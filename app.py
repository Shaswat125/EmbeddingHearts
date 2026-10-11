import chromadb
from sentence_transformers import SentenceTransformer
from constants import DB_PATH, MODEL_NAME, COLLECTION_NAME, PROFILES_FILE
import json
import os



def load_profiles_from_file(file_path=PROFILES_FILE):
    """Loads and validates dating profiles from an external JSON file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target data file '{file_path}' not found. Please ensure it exists.")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)
            print(f"Successfully loaded {len(profiles)} profiles from '{file_path}'.")
            return profiles
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from '{file_path}': {e}")
        return []
    

def initialize_profile_database(db_path=DB_PATH, model_name=MODEL_NAME, collection_name=COLLECTION_NAME, batch_size=100):
    """
    Initializes ChromaDB and synchronizes profile embeddings robustly under high load.
    """
    print("Initializing embedding runtime and vector storage engine...")
    model = SentenceTransformer(model_name, local_files_only=True)
    client = chromadb.PersistentClient(path=db_path)
    
    # Configure collection space for semantic angular alignment
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    all_dating_profiles = load_profiles_from_file()
    if not all_dating_profiles:
        print("No profiles provided for synchronization.")
        return

    total_profiles = len(all_dating_profiles)
    print(f"Synchronizing {total_profiles} profiles in batches of {batch_size}...")

    # Process ingestion in controlled chunks to manage memory overhead and payload limits
    for i in range(0, total_profiles, batch_size):
        batch = all_dating_profiles[i:i + batch_size]
        
        ids = [p["user_id"] for p in batch]
        documents = [p["bio"] for p in batch]
        metadatas = [{
            "name": p["name"],
            "age": p["age"],
            "gender": p["gender"],
            "interested_in": p["interested_in"],
            "location": p["location"]
        } for p in batch]

        # Generate normalized vector representations for length-agnostic comparisons
        embeddings = model.encode(documents, normalize_embeddings=True).tolist()

        # Execute upsert to handle both insertions and updates across existing states
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

    print(f"Successfully synchronized {total_profiles} profiles into collection '{collection_name}'.")


def search_dating_matches(user_id: str, n_results: int = 2, collection_name: str = COLLECTION_NAME):
    """
    Performs semantic matching for a given user:
    1. Fetches the user's bio and target preference criteria.
    2. Applies hard metadata filtering (e.g., matching gender preference).
    3. Runs vector similarity search on the remaining candidates.
    """
    model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(name=collection_name)

    target_record = collection.get(ids=[user_id], include=["documents", "metadatas"])
    if not target_record["ids"]:
        print(f"Error: User ID '{user_id}' not found.")
        return

    target_bio = target_record["documents"][0]
    target_meta = target_record["metadatas"][0]
    target_preference = target_meta["interested_in"]

    print(f"\n==================================================")
    print(f"Finding matches for: {target_meta['name']} ({target_meta['gender']}, looking for {target_preference})")
    print(f"Bio: '{target_bio}'")
    print(f"==================================================")

    # Format query using BGE instruction template
    query_instruction = f"Represent this query for searching relevant passages: {target_bio}"
    query_embedding = model.encode([query_instruction], normalize_embeddings=True).tolist()

    # Query ChromaDB with vector + metadata filter (exclude self and match preference)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results + 1,  # Request extra to handle self-filtering
        where={"gender": {"$eq": target_preference}}
    )

    if not results or not results["ids"] or not results["ids"][0]:
        print("No matching profiles found with current filters.")
        return

    match_count = 0
    for i, (uid, doc, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ):
        # Skip if the result is the user themselves
        if uid == user_id:
            continue

        similarity_score = max(0.0, 1.0 - distance)
        match_count += 1

        print(f"\nMatch [{match_count}] -> {metadata['name']} ({metadata['age']}, {metadata['location']})")
        print(f"Similarity Score : {similarity_score:.4f} ({similarity_score * 100:.1f} compatibility)")
        print(f"Bio              : {doc}")


if __name__ == "__main__":
    # 1. Initialize and load sample data
    # initialize_profile_database(db_path=DB_PATH, model_name=MODEL_NAME, collection_name=COLLECTION_NAME, batch_size=100)

    # 2. Test search for Aarav (u_001) looking for female profiles with similar lifestyle/interests
    search_dating_matches(user_id="u_001", n_results=2, collection_name=COLLECTION_NAME)

    # 3. Test search for Ananya (u_002) looking for male profiles
    search_dating_matches(user_id="u_002", n_results=2, collection_name=COLLECTION_NAME)