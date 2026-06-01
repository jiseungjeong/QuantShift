from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/layerskip-llama2-7B")
model = AutoModelForCausalLM.from_pretrained("facebook/layerskip-llama2-7B")