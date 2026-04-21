import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

collection = None
embedding_model = None

def init_retrieval():
    global collection, embedding_model
    client = chromadb.PersistentClient(path="data/chroma_db")
    collection = client.get_collection("support_tickets_chunks")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Retrieval system initialized.")

def _convert_to_python(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_python(i) for i in obj]
    else:
        return obj

def retrieve_similar(query: str, top_k: int = 5):
    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for i in range(len(results['documents'][0])):
        metadata = _convert_to_python(results['metadatas'][0][i])
        chunks.append({
            "text": results['documents'][0][i],
            "metadata": metadata,
            "similarity": float(1 - results['distances'][0][i])
        })
    return chunks