import sys, os, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from utils import get_device, sync
from engine.model import GPT2
from engine.batch import generate_static
from engine.continuous import Request, generate_continuous

device = get_device()
assert device.type == "cuda", "CUDA graphs need an NVIDIA GPU"
tok = AutoTokenizer.from_pretrained("gpt2")
model = GPT2.from_pretrained("gpt2").to(device)

texts = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "Hi",
    "When you are holding pocket aces before the flop, you should usually",
]
pool = [tok(t, return_tensors="pt").input_ids.to(device) for t in texts]
modes = ["dynamic", "static", "graph"]
N = 200

def timed(fn):
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - start

print("Static batching, 200 tokens per sequence (tok/s)")
print(f"{'batch':>5} | {'dynamic':>8} | {'static':>8} | {'graph':>8} | {'graph vs dyn':>12}")
for B in [1, 8, 16]:
    prompts = [pool[i % len(pool)] for i in range(B)]
    tps = {}
    for mode in modes:
        generate_static(model, prompts, 10, mode=mode)
        t = timed(lambda: generate_static(model, prompts, N, mode=mode))
        tps[mode] = B * N / t
    print(f"{B:5d} | {tps['dynamic']:8.1f} | {tps['static']:8.1f} | "
          f"{tps['graph']:8.1f} | {tps['graph'] / tps['dynamic']:11.2f}x")

N_REQ, N_SLOTS = 32, 8
rng = random.Random(0)
lengths = [rng.randint(10, 200) for _ in range(N_REQ)]
prompts = [pool[i % len(pool)] for i in range(N_REQ)]
useful = sum(lengths)

print(f"\nContinuous batching, {N_REQ} requests, {N_SLOTS} slots (useful tok/s)")
base = None
for mode in modes:
    run = lambda: generate_continuous(
        model, [Request(i, prompts[i], lengths[i]) for i in range(N_REQ)],
        N_SLOTS, mode=mode)
    generate_continuous(model, [Request(0, pool[0], 10)], N_SLOTS, mode=mode)
    tps = useful / timed(run)
    base = base or tps
    print(f"{mode:>8} | {tps:8.1f} | {tps / base:5.2f}x")
