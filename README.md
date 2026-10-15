# nano-infer

A GPT-2 inference engine I wrote from scratch in PyTorch to learn how LLM serving works. It has a KV cache, continuous batching, speculative decoding, INT8 weight quantization, and CUDA graphs, and I benchmarked each of them on an NVIDIA T4.

The model code, attention, cache, scheduler, and decoding loops are mine. HuggingFace is only used to download the weights and as a reference to check my outputs against.

## Results

T4, PyTorch 2.11, fp32 unless noted, greedy decoding. Numbers are medians of repeated runs.

| | result |
|---|---|
| Single-sequence decode, GPT-2 small | 88 to 404 tok/s in fp32 (4.6x), 629 tok/s with INT8 (7.2x) |
| vs. HF `generate()`, batch 1 | 3.9x faster than default, 1.57x faster than static cache + compile |
| vs. HF `generate()`, batch 16 | 1.49x faster than default, same as static cache |
| 32 mixed-length requests | 2.35x the throughput of static batching |
| Speculative decoding, GPT-2 large | 2.53x, same output as the target alone |
| INT8 per-channel | half the memory, no perplexity change |

The full breakdown, what I learned from each stage, and a debugging story about torch.compile are in [NOTES.md](NOTES.md). Raw data is in `results/`.

## What's in here

| component | file |
|---|---|
| GPT-2 forward pass, loading HF weights | `engine/model.py` |
| Naive and KV-cached generation | `engine/naive.py`, `engine/kv.py` |
| Slot KV cache, batched / CUDA graph / compiled decode | `engine/batch.py` |
| Continuous batching scheduler | `engine/continuous.py` |
| Speculative decoding | `engine/speculative.py` |
| INT8 quantization | `engine/quant.py` |

Every engine is tested against HF greedy decoding token for token. Batched, continuous, and speculative outputs also have to match single-sequence output exactly, including when a slot gets reused mid-generation. INT8 changes the weights, so I checked it with perplexity instead.

## Running it

```bash
pip install -r requirements.txt   # assumes PyTorch with CUDA is already installed
bash run_tests.sh                 # correctness tests, CPU or GPU
python bench/final.py             # main results, about 15 min on a T4
python bench/compare_hf.py        # HF comparison
```

I ran everything GPU-related on Google Colab with a T4.
