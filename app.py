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


def search_dating_matches(
    user_id: str, 
    n_results: int = 3, 
    age_window: int = 15, 
    min_age_override: int = None, 
    max_age_override: int = None,
    collection_name: str = COLLECTION_NAME,
    db_path: str = DB_PATH,
    model_name: str = MODEL_NAME
):
    """
    Performs comprehensive hybrid search (vector similarity + composite metadata filters):
    1. Fetches target user record to extract profile bio, age, and orientation.
    2. Constructs strict hard filters for gender preference and age boundaries.
    3. Executes vector similarity search on the filtered candidate subset.
    """
    print(f"Initializing embedding runtime and vector storage for query...")
    model = SentenceTransformer(model_name, local_files_only=True)
    client = chromadb.PersistentClient(path=db_path)
    
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error accessing collection '{collection_name}': {e}")
        return

    # 1. Fetch querying user metadata and text
    target_record = collection.get(ids=[user_id], include=["documents", "metadatas"])
    if not target_record["ids"]:
        print(f"Error: User ID '{user_id}' not found in the vector database.")
        return

    target_bio = target_record["documents"][0]
    target_meta = target_record["metadatas"][0]
    target_gender = target_meta["gender"]
    target_preference = target_meta["interested_in"]
    target_age = target_meta["age"]

    # 2. Determine age boundaries (use overrides if provided, otherwise compute dynamic window)
    min_age = min_age_override if min_age_override is not None else max(18, target_age - age_window)
    max_age = max_age_override if max_age_override is not None else target_age + age_window

    print(f"\n==================================================")
    print(f"Match Request: {target_meta['name']} ({target_age} yrs, {target_gender})")
    print(f"Target Filters -> Looking for: {target_preference}, Age Range: {min_age}-{max_age}")
    print(f"==================================================")

    # 3. Format query using BGE instruction template
    query_instruction = f"Represent this query for searching relevant passages: {target_bio}"
    query_embedding = model.encode([query_instruction], normalize_embeddings=True).tolist()

    # 4. Construct composite metadata filter using ChromaDB operators ($and, $eq, $gte, $lte)
    metadata_filter = {
        "$and": [
            {"gender": {"$eq": target_preference}},
            {"age": {"$gte": min_age}},
            {"age": {"$lte": max_age}}
        ]
    }

    # 5. Query ChromaDB with vector embeddings and strict metadata constraints
    # Request extra results to safely account for self-exclusion post-filtering
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results + 5,
        where=metadata_filter
    )

    if not results or not results["ids"] or not results["ids"][0]:
        print("No matching profiles found satisfying the specified age and gender criteria.")
        return

    match_count = 0
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for uid, doc, metadata, distance in zip(ids, documents, metadatas, distances):
        # Exclude the querying user from their own recommendations
        if uid == user_id:
            continue

        # Convert cosine distance to compatibility percentage score
        similarity_score = max(0.0, 1.0 - distance)
        match_count += 1

        print(f"\nMatch [{match_count}] -> {metadata['name']} ({metadata['age']} yrs, {metadata['location']})")
        print(f"Compatibility Score : {similarity_score:.4f} ({similarity_score * 100:.1f}%)")
        print(f"Candidate Bio       : {doc}")

        if match_count >= n_results:
            break

    if match_count == 0:
        print("No external matches found after filtering out self-profile.")


if __name__ == "__main__":
    # 1. Initialize and load sample data
    # initialize_profile_database(db_path=DB_PATH, model_name=MODEL_NAME, collection_name=COLLECTION_NAME, batch_size=100)

    # 2. Test search for Aarav (u_001) looking for female profiles with similar lifestyle/interests
    search_dating_matches(user_id="u_001", n_results=2, collection_name=COLLECTION_NAME)

    # 3. Test search for Ananya (u_002) looking for male profiles
    search_dating_matches(user_id="u_002", n_results=2, collection_name=COLLECTION_NAME)