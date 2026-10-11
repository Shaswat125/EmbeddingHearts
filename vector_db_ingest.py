import json
import os
import chromadb

from sentence_transformers import SentenceTransformer

# Paths
DB_PATH = "./vector_db"
DATA_FILE = "data.json"

# Lightweight embedding model (~133 MB model footprint)
MODEL_NAME = "BAAI/bge-small-en-v1.5"
# Alternative (~90 MB option): MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

COLLECTION_NAME = "huggingface_cosine_collection"


def initialize_database():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found. Please create it first.")
        return

    print(f"Loading lightweight Hugging Face embedding model: {MODEL_NAME}...")
    # BGE models benefit from an instruction prefix for retrieval tasks
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)

    print(f"Connecting to local Chroma vector database at '{DB_PATH}'...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Configuring ChromaDB to use 'cosine' distance space
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    # Load raw text data from JSON
    print(f"Loading documents from {DATA_FILE}...")
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids = [item["id"] for item in data]
    
    # Prepend prompt instruction for BGE models
    raw_texts = [item["text"] for item in data]
    texts_to_embed = [
        f"Represent this document for retrieval: {t}" for t in raw_texts
    ]
    metadatas = [{"category": item["category"], "text": item["text"]} for item in data]

    print("Generating dense vector embeddings...")
    embeddings = model.encode(texts_to_embed, normalize_embeddings=True).tolist()

    # Upsert records into the local vector collection
    collection.upsert(
        ids=ids, embeddings=embeddings, metadatas=metadatas, documents=raw_texts
    )

    print(
        f"Successfully indexed {len(ids)} documents into ChromaDB with Cosine metric!"
    )


if __name__ == "__main__":
    initialize_database()