import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from utils import get_device, sync
from engine.model import GPT2
from engine.kv import generate_kv
from engine.speculative import generate_speculative

TARGET = sys.argv[1] if len(sys.argv) > 1 else "gpt2-medium"
device = get_device()
modes = ["tuple", "graph"] if device.type == "cuda" else ["tuple"]
print("device:", device, "| target:", TARGET, "| draft: gpt2")
tok = AutoTokenizer.from_pretrained("gpt2")
target = GPT2.from_pretrained(TARGET).to(device)
draft = GPT2.from_pretrained("gpt2").to(device)

texts = [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "The history of the Roman Empire began when",
    "To train a neural network, you first need",
]
prompts = [tok(t, return_tensors="pt").input_ids.to(device) for t in texts]
N = 100

def timed(fn):
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - start

generate_kv(target, prompts[0], 10)
base_tps = len(prompts) * N / timed(lambda: [generate_kv(target, p, N) for p in prompts])
print(f"\ntarget alone: {base_tps:.1f} tok/s")

for mode in modes:
    generate_speculative(target, draft, prompts[0], 10, draft_mode=mode)
    print(f"\ndraft_mode={mode}")
    print(f"{'n_draft':>7} | {'accept':>6} | {'tok/s':>6} | {'speedup':>7}")
    for n_draft in range(1, 9):
        totals = {"proposed": 0, "accepted": 0}
        def run():
            for p in prompts:
                stats = {}
                generate_speculative(target, draft, p, N, n_draft=n_draft,
                                     stats=stats, draft_mode=mode)
                totals["proposed"] += stats["proposed"]
                totals["accepted"] += stats["accepted"]
        tps = len(prompts) * N / timed(run)
        rate = totals["accepted"] / totals["proposed"]
        print(f"{n_draft:7d} | {rate:6.0%} | {tps:6.1f} | {tps / base_tps:6.2f}x")
