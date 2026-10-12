import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_device
from engine.naive import generate_naive

device = get_device()
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()

ids = tok("The best poker players know that", return_tensors="pt").input_ids.to(device)

ref = model.generate(ids, max_new_tokens=50, do_sample=False,
                     pad_token_id=tok.eos_token_id)
ours = generate_naive(model, ids, 50)

assert torch.equal(ref, ours), "Mismatch!"
print("PASS")
print(tok.decode(ours[0]))
