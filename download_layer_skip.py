import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "facebook/layerskip-llama2-7B"

# directory to store the model inside the project tree
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(PROJECT_DIR, "models", MODEL_ID.split("/")[-1])
os.makedirs(LOCAL_DIR, exist_ok=True)

# download (cache_dir pointed inside the project)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=LOCAL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=LOCAL_DIR)

# also save as a clean local folder (can be loaded directly from LOCAL_DIR later)
tokenizer.save_pretrained(LOCAL_DIR)
model.save_pretrained(LOCAL_DIR)

print(f"Model saved to: {LOCAL_DIR}")
