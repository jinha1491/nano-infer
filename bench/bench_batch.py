import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from utils import get_device, sync
from engine.model import GPT2
from engine.kv import generate_kv
from engine.batch import generate_static

device = get_device()
print("device:", device)
tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2.from_pretrained("gpt2").to(device)

texts = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "Hi",
    "When you are holding pocket aces before the flop, you should usually",
]
pool = [tok(t, return_tensors="pt").input_ids.to(device) for t in texts]
N_NEW = 100

def timed(fn):
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - start

generate_kv(model, pool[0], 10)
seq_time = timed(lambda: [generate_kv(model, p, N_NEW) for p in pool])
seq_tps = len(pool) * N_NEW / seq_time
print(f"\nsequential (one at a time): {seq_tps:.1f} tok/s")

print(f"\n{'batch':>5} | {'tok/s':>7} | {'speedup':>7}")
for B in [1, 4, 8, 16]:
    prompts = [pool[i % len(pool)] for i in range(B)]
    generate_static(model, prompts, 10)
    t = timed(lambda: generate_static(model, prompts, N_NEW))
    tps = B * N_NEW / t
    print(f"{B:5d} | {tps:7.1f} | {tps / seq_tps:6.1f}x")
