import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import get_device, sync
from engine.naive import generate_naive

device = get_device()
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").to(device).eval()
ids = tok("The best poker players know that", return_tensors="pt").input_ids.to(device)

generate_naive(model, ids, 10)

for n in [50, 200, 500]:
    sync(device)
    start = time.perf_counter()
    generate_naive(model, ids, n)
    sync(device)
    elapsed = time.perf_counter() - start
    print(f"{n:4d} tokens: {elapsed:6.2f} s  |  {n / elapsed:6.1f} tokens/sec")
