import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.profiler import profile, ProfilerActivity
from engine.model import GPT2
from engine.batch import SlotCache, model_decode, compiled_model_decode

model = GPT2.from_pretrained("gpt2").cuda()
cache = SlotCache(model, 16, 512)
idx = torch.zeros(16, 1, dtype=torch.long, device="cuda")

for name, fn in [("eager", model_decode), ("compiled", compiled_model_decode)]:
    for _ in range(3):
        fn(model, idx, cache, True)
        cache.lens.zero_()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            fn(model, idx, cache, True)
            cache.lens.zero_()
        torch.cuda.synchronize()

    print(f"\n===== {name}: 10 decode steps, batch 16 =====")
    print(prof.key_averages().table(sort_by="device_time_total", row_limit=10))
