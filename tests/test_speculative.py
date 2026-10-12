import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from engine.model import GPT2
from engine.kv import generate_kv
from engine.speculative import generate_speculative

tok = AutoTokenizer.from_pretrained("gpt2")
target = GPT2.from_pretrained("gpt2-medium")
draft = GPT2.from_pretrained("gpt2")

ids = tok("The best poker players know that", return_tensors="pt").input_ids
N = 40
ref = generate_kv(target, ids, N)

for n_draft in [1, 3, 5]:
    stats = {}
    out = generate_speculative(target, draft, ids, N, n_draft=n_draft, stats=stats)
    assert torch.equal(ref, out), f"n_draft={n_draft}: mismatch!"
    rate = stats["accepted"] / stats["proposed"]
    print(f"PASS n_draft={n_draft}: {stats['rounds']} target passes for {N} tokens, "
          f"acceptance {rate:.0%}")

print(tok.decode(ref[0]))
