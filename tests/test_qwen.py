import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from engine.qwen import Qwen
from engine.naive import generate_naive
from engine.kv import generate_kv

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-0.6B"
dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(name)
hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(dev).eval()
ours = Qwen.from_pretrained(name).to(dev)

ids = tok("The best poker players know that", return_tensors="pt").input_ids.to(dev)

with torch.no_grad():
    diff = (hf(ids).logits - ours(ids)[0]).abs().max().item()
print(f"max logit diff vs HF: {diff:.2e}")

ref = hf.generate(ids, max_new_tokens=50, do_sample=False,
                  pad_token_id=tok.eos_token_id)

for label, fn in {"naive": generate_naive, "kv_cache": generate_kv}.items():
    out = fn(ours, ids, 50)[:, :ref.size(1)]
    assert torch.equal(ref, out), f"{label}: mismatch!"
    print(f"PASS: {label}")
