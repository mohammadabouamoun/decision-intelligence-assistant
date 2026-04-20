import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import chromadb
from sentence_transformers import SentenceTransformer
import os
import pickle

# ------------------------------
# 0. Sample size (adjust as needed)
# ------------------------------
SAMPLE_SIZE = 5000   # Smaller for faster chunking

# ------------------------------
# 1. Load and clean data (sample)
# ------------------------------
print("Loading data...")
df = pd.read_csv('data/twcs/twcs.csv')
print(f"Original shape: {df.shape}")

if len(df) > SAMPLE_SIZE:
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
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
# 2. Priority labeling (same as before)
# ------------------------------
urgent_keywords = ['refund', 'broken', 'cancel', 'down', 'help', 'urgent', 'problem', 'issue', 'not working', 'error', 'complaint', 'charge', 'money', 'lost', 'stolen', 'urgently', 'asap', 'immediately', 'emergency', 'critical', 'frustrated', 'angry', 'disappointed']

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
# 3. Chunk text into smaller pieces
# ------------------------------
def chunk_text(text, chunk_size=200, overlap=50):
    """
    Split text into chunks of approximately `chunk_size` characters,
    with `overlap` characters between chunks.
    """
    text = str(text)
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

# Apply chunking to each tweet
df['chunks'] = df['text'].apply(chunk_text)

# Explode the chunks into separate rows (each chunk becomes a document)
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
print(f"Total chunks created: {len(chunk_df)}")

# ------------------------------
# 4. Generate embeddings and store in Chroma (for each chunk)
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
# 5. ML baseline (still on original tweets, using whole text features)
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

X = feature_df.drop('priority', axis=1)
y = feature_df['priority']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nML Baseline Performance (Random Forest):")
print(classification_report(y_test, y_pred))

os.makedirs('backend/models', exist_ok=True)
with open('backend/models/ml_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('backend/models/feature_columns.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)

print("ML model saved to backend/models/ml_model.pkl")
print("Data preparation complete (with chunking).")