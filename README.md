# nano-infer

LLM inference engine written from scratch in PyTorch. Runs Qwen3 with a paged KV cache, prefix caching, continuous batching, speculative decoding, INT8 weights, and CUDA graphs.

The model, attention, caches, scheduler, and decoding loops are my own code. HuggingFace is only used for the tokenizer, downloading weights, and as a reference for correctness tests.

## Features

- Qwen3 and Qwen2.5 (RoPE, grouped-query attention, QK-norm, SwiGLU), plus GPT-2
- Weights loaded directly from safetensors
- Paged KV cache with block tables and admission control
- Prefix caching with block hashing, reference counts, and LRU eviction
- Continuous batching
- CUDA graph and torch.compile decode
- Speculative decoding with a draft model
- INT8 weight-only quantization
- Per-request temperature and top-p sampling
- fp32 and fp16

## Usage

~~~python
import torch
from transformers import AutoTokenizer
from engine.qwen import Qwen
from engine.continuous import Request, generate_continuous

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = Qwen.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float16).cuda()

prompts = ["The best poker players know that", "Bluffing works best when"]
requests = [
    Request(i, tok(p, return_tensors="pt").input_ids.cuda(), max_new_tokens=50,
            temperature=0.7, top_p=0.9)
    for i, p in enumerate(prompts)
]
out = generate_continuous(model, requests, n_slots=4, n_blocks=256, prefix_cache=True)

for i in range(len(prompts)):
    print(tok.decode(out[i]))
~~~

Leave out `n_blocks` to use the simpler slot cache instead of paging.

## Results

Qwen3-0.6B, T4 on Colab, fp16, greedy decoding. Medians of repeated runs.

| benchmark | result |
|---|---|
| Single-sequence decode | 22 to 152 tok/s with CUDA graphs + torch.compile, 199 tok/s with INT8 |
| vs. HF `generate()`, batch 1 | 6.8x faster than default, 1.27x faster than static cache |
| Paged vs. slot cache, same KV memory | 1.78x throughput (15.6 vs. 7.2 concurrent requests) |
| Prefix caching, 1,384-token shared prompt | 2.56x faster, 95% fewer prompt tokens computed |
| Speculative decoding, Qwen3-1.7B + 0.6B draft | 1.74x, identical output to the target |

Full tables and explanations are in [NOTES.md](NOTES.md), including the earlier GPT-2 results. Raw data is in `results/`.

## Tests

Every decode path is checked against HuggingFace greedy decoding token for token in fp32. Batched, continuous, paged, prefix-cached, and speculative outputs must also match single-sequence output exactly, including when slots and blocks are reused mid-run. The paged cache asserts that every block is returned at the end of a run. fp16 and INT8 are checked with top-1 agreement and logit differences, since exact matches aren't expected there.

~~~bash
pip install -r requirements.txt        # assumes PyTorch with CUDA is installed
bash run_tests.sh                      # GPT-2 tests
python tests/test_qwen.py              # Qwen3 vs. HF
python tests/test_qwen_batch.py        # batched, continuous, CUDA graph
python tests/test_paged.py             # paged KV cache
python tests/test_prefix.py            # prefix caching
python tests/test_fp16.py              # fp16 (GPU)
python tests/test_qwen_spec_quant.py   # speculative decoding, INT8 (GPU)
python tests/test_sampling.py          # sampling
~~~

## Benchmarks

~~~bash
python bench/bench_qwen.py           # about 20 min on a T4
python bench/bench_prefix_long.py    # long-prompt prefix caching
~~~

## Project layout

~~~
engine/
  qwen.py          Qwen2.5 / Qwen3 model and weight loading
  model.py         GPT-2 model
  kv.py            single-sequence KV-cached generation
  batch.py         slot KV cache, batched / CUDA graph / compiled decode
  paged.py         paged KV cache and prefix caching
  continuous.py    continuous batching scheduler
  sampling.py      temperature and top-p sampling
  speculative.py   speculative decoding
  quant.py         INT8 quantization
tests/             correctness tests
bench/             benchmarks
results/           benchmark output
~~~

## Limitations

- Attention is plain PyTorch. The paged cache gathers blocks into a contiguous tensor every step instead of using a PagedAttention kernel, and there is no FlashAttention.
- INT8 dequantizes weights each step, so it only helps at small batch sizes.
- No preemption. Requests are admitted only when their worst-case length fits in the block pool.
- Speculative decoding is greedy only.
- Single GPU, and only tested on a T4.

## References

- Kwon et al., [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180), 2023
- Leviathan et al., [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), 2023
- [vLLM](https://github.com/vllm-project/vllm) and [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) for design ideas
