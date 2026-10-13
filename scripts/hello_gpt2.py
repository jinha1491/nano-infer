from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_device

device = get_device()
print("Using device:", device)

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

ids = tok("The best poker players know that", return_tensors="pt").input_ids.to(device)
print("Input token ids:", ids)

out = model.generate(ids, max_new_tokens=30, do_sample=False,
                     pad_token_id=tok.eos_token_id)
print(tok.decode(out[0]))
