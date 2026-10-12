import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from utils import get_device, sync
from engine.model import GPT2
from engine.naive import generate_naive
from engine.kv import generate_kv

device = get_device()
print("device:", device)
tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2.from_pretrained("gpt2").to(device)
ids = tok("The best poker players know that", return_tensors="pt").input_ids.to(device)

engines = {"naive": generate_naive, "kv_cache": generate_kv}
lengths = [50, 200, 500]
results = {}

for name, fn in engines.items():
    fn(model, ids, 10)
    for n in lengths:
        sync(device)
        start = time.perf_counter()
        fn(model, ids, n)
        sync(device)
        results[(name, n)] = n / (time.perf_counter() - start)

print(f"\n{'tokens':>6} | {'naive tok/s':>11} | {'kv tok/s':>8} | {'speedup':>7}")
for n in lengths:
    a, b = results[("naive", n)], results[("kv_cache", n)]
    print(f"{n:6d} | {a:11.1f} | {b:8.1f} | {b / a:6.1f}x")
