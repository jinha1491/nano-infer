import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from utils import get_device, sync
from engine.model import GPT2
from engine.batch import generate_static
from engine.continuous import Request, generate_continuous

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

N_REQ, N_SLOTS = 32, 8
rng = random.Random(0)
lengths = [rng.randint(10, 200) for _ in range(N_REQ)]
prompts = [pool[i % len(pool)] for i in range(N_REQ)]
useful = sum(lengths)

groups = [range(i, min(i + N_SLOTS, N_REQ)) for i in range(0, N_REQ, N_SLOTS)]
slot_steps = sum(len(g) * max(lengths[j] for j in g) for g in groups)
print(f"{N_REQ} requests, {N_SLOTS} slots, {useful} useful tokens")
print(f"static batching slot utilization: {useful / slot_steps:.0%}")

def run_static():
    for g in groups:
        generate_static(model, [prompts[j] for j in g], max(lengths[j] for j in g))

def run_continuous():
    reqs = [Request(i, prompts[i], lengths[i]) for i in range(N_REQ)]
    generate_continuous(model, reqs, N_SLOTS)

def timed(fn):
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - start

generate_static(model, pool, 10)
generate_continuous(model, [Request(i, p, 10) for i, p in enumerate(pool)], N_SLOTS)

t_static = timed(run_static)
t_cont = timed(run_continuous)

print(f"\n{'method':>10} | {'time (s)':>8} | {'useful tok/s':>12}")
print(f"{'static':>10} | {t_static:8.2f} | {useful / t_static:12.1f}")
print(f"{'continuous':>10} | {t_cont:8.2f} | {useful / t_cont:12.1f}")
print(f"\ncontinuous speedup: {t_static / t_cont:.2f}x")

stats = {}
generate_continuous(model, [Request(i, prompts[i], lengths[i]) for i in range(N_REQ)],
                    N_SLOTS, stats=stats)
steps = stats["decode_steps"]
min_steps = -(-(useful - N_REQ) // N_SLOTS)
print(f"\ncontinuous decode steps: {steps} (minimum possible: {min_steps})")
print(f"continuous slot utilization: {(useful - N_REQ) / (steps * N_SLOTS):.0%}")
print(f"static decode steps: {sum(max(lengths[j] for j in g) - 1 for g in groups)}")
