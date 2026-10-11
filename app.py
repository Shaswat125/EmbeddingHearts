import chromadb
from sentence_transformers import SentenceTransformer

DB_PATH = "./vector_db"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "huggingface_cosine_collection"


def semantic_search(query_text, n_results=2):
  print(f"\n==========================================")
  print(f"Query: '{query_text}'")
  print(f"==========================================")

  # 1. Load model and connect to vector store
  model = SentenceTransformer(MODEL_NAME, local_files_only=True)
  client = chromadb.PersistentClient(path=DB_PATH)
  collection = client.get_collection(name=COLLECTION_NAME)

  # 2. Embed the query (BGE instruction format for queries)
  query_instruction = f"Represent this query for searching relevant passages: {query_text}"
  query_embedding = model.encode(
      [query_instruction], normalize_embeddings=True
  ).tolist()

  # 3. Perform similarity search
  results = collection.query(
      query_embeddings=query_embedding, n_results=n_results
  )

  # 4. Display formatted results
  for i, (doc, metadata, distance) in enumerate(
      zip(
          results["documents"][0],
          results["metadatas"][0],
          results["distances"][0],
      )
  ):
    # For cosine metric in ChromaDB, distance = 1 - cosine_similarity
    # Convert distance back to a standard similarity score (0 to 1 scale)
    similarity_score = 1.0 - distance

    print(f"\nMatch [{i+1}] (Similarity Score: {similarity_score:.4f})")
    print(f"Category : {metadata['category']}")
    print(f"Content  : {doc}")


if __name__ == "__main__":
  # Run sample semantic queries (testing meaning over direct keyword hits)
  semantic_search("How do I build AI apps with autonomous agents?")
  semantic_search("Tips for keeping vehicle parts functioning properly")
  semantic_search("Python frameworks for building web APIs")