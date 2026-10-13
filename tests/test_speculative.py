import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer
from utils import get_device
from engine.model import GPT2
from engine.kv import generate_kv
from engine.speculative import generate_speculative

device = get_device() if torch.cuda.is_available() else torch.device("cpu")
tok = AutoTokenizer.from_pretrained("gpt2")
target = GPT2.from_pretrained("gpt2-medium").to(device)
draft = GPT2.from_pretrained("gpt2").to(device)

ids = tok("The best poker players know that", return_tensors="pt").input_ids.to(device)
N = 40
ref = generate_kv(target, ids, N)

modes = ["tuple", "dynamic", "static"]
if device.type == "cuda":
    modes.append("graph")

for mode in modes:
    for n_draft in [1, 3, 5]:
        stats = {}
        out = generate_speculative(target, draft, ids, N, n_draft=n_draft,
                                   stats=stats, draft_mode=mode, max_len=128)
        assert torch.equal(ref, out), f"{mode}, n_draft={n_draft}: mismatch!"
    print(f"PASS: draft_mode={mode}, n_draft 1/3/5")

print(tok.decode(ref[0]))
