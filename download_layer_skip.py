import os
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "facebook/layerskip-llama2-7B"

# 프로젝트 경로 안에 모델을 저장할 디렉토리
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DIR = os.path.join(PROJECT_DIR, "models", MODEL_ID.split("/")[-1])
os.makedirs(LOCAL_DIR, exist_ok=True)

# 다운로드 (cache_dir를 프로젝트 안으로 지정)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=LOCAL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, cache_dir=LOCAL_DIR)

# 깔끔하게 로컬 폴더 형태로도 저장 (나중에 LOCAL_DIR 경로로 바로 로드 가능)
tokenizer.save_pretrained(LOCAL_DIR)
model.save_pretrained(LOCAL_DIR)

print(f"모델이 저장된 경로: {LOCAL_DIR}")
