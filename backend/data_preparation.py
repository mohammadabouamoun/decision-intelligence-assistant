import pandas as pd
import numpy as np
import re
import random
from sentence_transformers import SentenceTransformer
import chromadb
import os

# ------------------------------
# 0. Parameters
# ------------------------------
SAMPLE_SIZE = 50000           # Number of inbound tweets to sample
CHUNK_SIZE = 500              # Large enough to fit whole tweet (max 280 chars)
OVERLAP = 0                   # No overlap needed
RANDOM_STATE = 42

def reservoir_sample_csv(file_path, sample_size, inbound_only=True, random_state=42):
    """
    Reservoir sampling from a CSV, reading in chunks.
    Optionally filter by inbound column (assumed to be boolean/0/1).
    """
    random.seed(random_state)
    sample = []
    total_processed = 0
    
    # Determine if inbound column exists by reading first chunk
    first_chunk = next(pd.read_csv(file_path, chunksize=1))
    has_inbound = 'inbound' in first_chunk.columns
    
    for chunk in pd.read_csv(file_path, chunksize=10000):
        if inbound_only and has_inbound:
            chunk = chunk[chunk['inbound'] == True]
        if chunk.empty:
            continue
        for _, row in chunk.iterrows():
            total_processed += 1
            if len(sample) < sample_size:
                sample.append(row)
            else:
                j = random.randint(0, total_processed - 1)
                if j < sample_size:
                    sample[j] = row
    if sample:
        return pd.DataFrame(sample)
    else:
        return pd.DataFrame()

# ------------------------------
# 1. Load and sample (memory efficient)
# ------------------------------
print("Loading data using reservoir sampling (inbound only)...")
df = reservoir_sample_csv('data/twcs/twcs.csv', SAMPLE_SIZE, inbound_only=True, random_state=RANDOM_STATE)
print(f"Sampled shape: {df.shape}")

df = df.dropna(subset=['text'])
print(f"After dropping null text: {df.shape}")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

# ------------------------------
# 2. Priority labeling (weak supervision)
# ------------------------------
urgent_keywords = ['refund', 'broken', 'cancel', 'down', 'help', 'urgent', 'problem', 'issue', 
                   'not working', 'error', 'complaint', 'charge', 'money', 'lost', 'stolen', 
                   'urgently', 'asap', 'immediately', 'emergency', 'critical', 'frustrated', 
                   'angry', 'disappointed']

def has_exclamation(text):
    return text.count('!') >= 2

def has_allcaps(text):
    words = text.split()
    if not words:
        return False
    upper_count = sum(1 for w in words if w.isupper())
    return upper_count / len(words) > 0.5

def is_urgent(row):
    text = row['text'].lower()
    if any(kw in text for kw in urgent_keywords):
        return 1
    if has_exclamation(row['text']):
        return 1
    if has_allcaps(row['text']):
        return 1
    return 0

df['priority'] = df.apply(is_urgent, axis=1)
print(f"Priority distribution:\n{df['priority'].value_counts(normalize=True)}")

# ------------------------------
# 3. Chunk text (one chunk per tweet, since CHUNK_SIZE > tweet length)
# ------------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    text = str(text)
    # If text is short, just return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
        if start >= len(text):
            break
    return chunks

df['chunks'] = df['text'].apply(chunk_text)

# Explode chunks into separate rows
chunked_rows = []
for idx, row in df.iterrows():
    for chunk_idx, chunk in enumerate(row['chunks']):
        chunked_rows.append({
            'chunk_id': f"{row['tweet_id']}_{chunk_idx}",
            'original_tweet_id': row['tweet_id'],
            'text': chunk,
            'priority': row['priority']
        })

chunk_df = pd.DataFrame(chunked_rows)
print(f"Total chunks created: {len(chunk_df)} (should equal number of tweets)")

# ------------------------------
# 4. Generate embeddings and store in Chroma
# ------------------------------
print("Loading embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

chroma_client = chromadb.PersistentClient(path="data/chroma_db")
try:
    chroma_client.delete_collection("support_tickets_chunks")
except:
    pass
collection = chroma_client.create_collection(name="support_tickets_chunks")

batch_size = 500
for i in range(0, len(chunk_df), batch_size):
    batch = chunk_df.iloc[i:i+batch_size]
    ids = batch['chunk_id'].tolist()
    texts = batch['text'].tolist()
    embeddings = embedding_model.encode(texts).tolist()
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"original_tweet_id": row['original_tweet_id'], "priority": int(row['priority'])} for _, row in batch.iterrows()]
    )
    print(f"Added batch {i//batch_size + 1} / {len(chunk_df)//batch_size + 1}")

print("Chroma database populated with chunks.")

# ------------------------------
# 5. Prepare tabular dataset for ML baseline
# ------------------------------
def extract_features(text):
    text = str(text)
    features = {
        'length': len(text),
        'word_count': len(text.split()),
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'caps_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'urgent_keyword_count': sum(1 for kw in urgent_keywords if kw in text.lower()),
        'has_urgent_keyword': int(any(kw in text.lower() for kw in urgent_keywords))
    }
    return features

feature_df = df['text'].apply(extract_features).apply(pd.Series)
feature_df['priority'] = df['priority']

os.makedirs('data', exist_ok=True)
feature_df.to_csv('data/features.csv', index=False)
print("Feature dataset saved to data/features.csv")
print("Data preparation complete. Run 'python backend/train_models.py' to train and compare models.")