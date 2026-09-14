from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("moonshotai/Kimi-Linear-48B-A3B-Base", trust_remote_code=True, device_map="auto")