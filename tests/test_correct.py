import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.model import GPT2
from engine.naive import generate_naive
from engine.kv import generate_kv

tok = AutoTokenizer.from_pretrained("gpt2")
hf = AutoModelForCausalLM.from_pretrained("gpt2").eval()
ours = GPT2.from_pretrained("gpt2")

ids = tok("The best poker players know that", return_tensors="pt").input_ids
ref = hf.generate(ids, max_new_tokens=50, do_sample=False,
                  pad_token_id=tok.eos_token_id)

engines = {"naive": generate_naive, "kv_cache": generate_kv}
for name, fn in engines.items():
    out = fn(ours, ids, 50)
    assert torch.equal(ref, out), f"{name}: mismatch!"
    print(f"PASS: {name}")
