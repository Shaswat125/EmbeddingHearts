from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
LOCAL_MODEL_PATH = "./local_bge_model"

print(f"Downloading {MODEL_NAME} from Hugging Face...")
model = SentenceTransformer(MODEL_NAME)

print(f"Saving model locally to {LOCAL_MODEL_PATH}...")
model.save(LOCAL_MODEL_PATH)
print("Model saved successfully! You can now run completely offline.")