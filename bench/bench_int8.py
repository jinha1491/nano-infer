import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from utils import get_device, sync
from engine.model import GPT2
from engine.batch import generate_static
from engine.quant import quantize_model, model_bytes

device = get_device()
assert device.type == "cuda", "needs an NVIDIA GPU"
tok = AutoTokenizer.from_pretrained("gpt2")
pool = [tok(t, return_tensors="pt").input_ids.to(device) for t in [
    "The best poker players know that",
    "In a tournament, the most important skill is",
    "Hi",
    "When you are holding pocket aces before the flop, you should usually",
]]
N = 200

def timed(fn):
    sync(device)
    start = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - start

models = {
    "fp32": GPT2.from_pretrained("gpt2").to(device),
    "int8": quantize_model(GPT2.from_pretrained("gpt2")).to(device),
}
for name, m in models.items():
    print(f"{name} memory: {model_bytes(m) / 1e6:.0f}MB")

print(f"\n{'batch':>5} | {'weights':>7} | {'graph':>7} | {'compiled':>8}   (tok/s)")
for B in [1, 16]:
    prompts = [pool[i % len(pool)] for i in range(B)]
    for name, m in models.items():
        row = {}
        for mode in ["graph", "compiled"]:
            generate_static(m, prompts, 10, mode=mode)
            row[mode] = B * N / timed(lambda: generate_static(m, prompts, N, mode=mode))
        print(f"{B:5d} | {name:>7} | {row['graph']:7.1f} | {row['compiled']:8.1f}")
