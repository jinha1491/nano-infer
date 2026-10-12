import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from engine.model import GPT2
from engine.kv import generate_kv
from engine.continuous import Request, generate_continuous

tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2.from_pretrained("gpt2")

texts = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "Hi",
    "When you are holding pocket aces before the flop, you should usually",
    "Bluffing works best when",
    "The river card changed everything because",
]
lengths = [5, 40, 12, 30, 1, 25]
requests = [Request(i, tok(t, return_tensors="pt").input_ids, n)
            for i, (t, n) in enumerate(zip(texts, lengths))]

for n_slots in [2, 3]:
    results = generate_continuous(model, [
        Request(r.rid, r.prompt, r.max_new_tokens) for r in requests], n_slots)
    for r in requests:
        ref = generate_kv(model, r.prompt, r.max_new_tokens)[0]
        assert torch.equal(ref, results[r.rid]), f"slots={n_slots} req {r.rid}: mismatch!"
    print(f"PASS: n_slots={n_slots}, all {len(requests)} requests match")
