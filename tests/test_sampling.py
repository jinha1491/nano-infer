import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
from engine.sampling import sample

def freqs(logits, temp, top_p, n=20000):
    g = torch.Generator().manual_seed(0)
    L = logits.repeat(n, 1)
    out = sample(L, torch.full((n,), temp), torch.full((n,), top_p), g)
    return torch.bincount(out[:, 0], minlength=logits.size(-1)).float() / n

logits = torch.tensor([[math.log(0.7), math.log(0.3)]])
f = freqs(logits, 1.0, 1.0)
assert abs(f[0] - 0.7) < 0.02, f
print(f"PASS: temperature 1 keeps the distribution ({f[0]:.3f} vs 0.700)")

f = freqs(logits, 2.0, 1.0)
expect = 0.7 ** 0.5 / (0.7 ** 0.5 + 0.3 ** 0.5)
assert abs(f[0] - expect) < 0.02, f
print(f"PASS: temperature 2 flattens it ({f[0]:.3f} vs {expect:.3f})")

f = freqs(logits, 1.0, 0.5)
assert f[0] == 1.0, f
print("PASS: top_p=0.5 keeps only the top token")

rand = torch.randn(8, 1000)
zeros, ones = torch.zeros(8), torch.ones(8)
assert torch.equal(sample(rand, zeros, ones), rand.argmax(-1, keepdim=True))
assert torch.equal(sample(rand, ones, torch.full((8,), 1e-6)), rand.argmax(-1, keepdim=True))
print("PASS: temperature 0 and tiny top_p both reduce to greedy")

mixed = sample(rand, torch.tensor([0.0, 1.0] * 4), ones, torch.Generator().manual_seed(1))
assert torch.equal(mixed[0::2], rand[0::2].argmax(-1, keepdim=True))
print("PASS: greedy rows stay greedy inside a mixed batch")

if not torch.cuda.is_available():
    print("SKIP: model tests need a GPU")
    sys.exit(0)

from transformers import AutoTokenizer
from engine.qwen import Qwen
from engine.kv import generate_kv
from engine.continuous import Request, generate_continuous

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = Qwen.from_pretrained("Qwen/Qwen3-0.6B").cuda()
texts = ["The best poker players know that", "In a tournament, the most important skill is",
         "Bluffing works best when", "The river card changed everything because"]
prompts = [tok(t, return_tensors="pt").input_ids.cuda() for t in texts]
refs = [generate_kv(model, p, 30)[0].cpu() for p in prompts]

def run(seed):
    reqs = [Request(i, p, 30, temperature=0.0 if i % 2 == 0 else 0.8, top_p=0.9)
            for i, p in enumerate(prompts)]
    return generate_continuous(model, reqs, 3, max_len=128, n_blocks=64, seed=seed)

a, b, c = run(0), run(0), run(1)
for i in [0, 2]:
    assert torch.equal(a[i], refs[i]), f"greedy req {i} changed"
print("PASS: greedy requests still match exactly while sharing a batch with sampled ones")
assert all(torch.equal(a[i], b[i]) for i in range(4))
print("PASS: same seed gives identical samples")
print("different seed changes the sampled requests:",
      any(not torch.equal(a[i], c[i]) for i in [1, 3]))
for i in [1, 3]:
    print(f"  sampled {i}:", repr(tok.decode(a[i][prompts[i].size(1):])))
