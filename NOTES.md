# Notes

## Qwen3 version

All results in this section are Qwen3-0.6B on a T4, fp16, greedy decoding, medians of repeated runs. The GPT-2 notes are further down.

### Porting to Qwen3

Moving from GPT-2 meant implementing RoPE, RMSNorm, SwiGLU, grouped-query attention, and Qwen3's QK-norm. Qwen3-0.6B has 16 query heads but 8 KV heads, so the cache stores half as many heads. In decode I reshape q into groups instead of repeating K and V for every query head, so the cache is never copied. Qwen3 also sets its head size separately from the hidden size (128 wide heads with a hidden size of 1024), which broke my first version because I assumed `d_head = d // n_head`.

The first loader built a full HF model and copied its weights over, which needed two copies in memory and stalled Colab on Qwen3-1.7B. It now reads the safetensors files directly, one tensor at a time.

### Single-sequence decode

200 new tokens.

| engine | tok/s | vs. HF default |
|---|---|---|
| KV cache | 22 | 0.97x |
| CUDA graph | 83 | 3.74x |
| + torch.compile | 152 | 6.84x |
| + INT8 | 199 | 8.94x |
| HF `generate()` default | 22 | 1.00x |
| HF `generate()` static cache | 120 | 5.40x |

Qwen3 has 28 layers, so every token launches hundreds of small kernels. The plain KV cache is completely launch-bound and ties HF's default. CUDA graphs alone give 3.7x.

### Batched decode

Total tok/s, 200 new tokens per sequence.

| batch | CUDA graph | compiled | compiled INT8 |
|---|---|---|---|
| 1 | 80 | 150 | 195 |
| 8 | 492 | 718 | 494 |
| 16 | 749 | 979 | 743 |

Same pattern as GPT-2: INT8 wins at batch 1, where decode is memory-bound, and loses at batch 8 and 16, where converting the weights back to fp16 every step costs more than the bandwidth it saves.

INT8 in fp16 first produced NaNs. I applied the scale after the matmul, so outputs were up to 127x too large before scaling, and Qwen's MLP activations pushed that past fp16's max of 65,504. In fp16 the weights are now dequantized before the matmul. fp32 keeps the scale-after version.

### Paged KV cache

64 requests with 16 to 256 new tokens, both caches given 4,096 tokens of KV memory.

| cache | useful tok/s | avg concurrent requests |
|---|---|---|
| slot cache, 8 slots of 512 | 132 | 7.2 |
| paged, 256 blocks of 16 | 234 | 15.6 |

The slot cache reserves `max_len` for every request, so most of its memory sits empty. The paged cache hands out 16-token blocks as sequences grow. A request is admitted only if the pool can cover its worst case, so nothing ever runs out of blocks mid-generation and there's no need for preemption.

Reading a paged cache in plain PyTorch means gathering each sequence's blocks into a contiguous tensor every step, which is a copy. vLLM avoids it with a custom attention kernel that reads blocks in place. That copy is also why I benchmarked paging in dynamic mode: under a CUDA graph, every slot would gather its full `max_len` every step.

### Prefix caching

32 requests sharing a system prompt, 16 new tokens each.

| shared prompt | prompt tokens computed (off / on) | speedup |
|---|---|---|
| 280 tokens | 9,334 / 902 | 1.05x |
| 1,384 tokens | 44,662 / 2,006 | 2.56x |

Each full block of prompt tokens is keyed by a SHA-256 hash of the previous block's key plus its own tokens, so a block only matches if everything before it matched too. Shared blocks are reference counted. When nobody is using a cached block, it stays in an LRU list until the pool needs the space, so a prompt can be reused even after the request that computed it has finished. Blocks are released last-first, so a chain's root survives longest.

With the 280-token prompt, prefill was launch-bound: one forward pass over 280 tokens takes about as long as one over 30, so cutting 90% of the math barely changed wall time. At 1,384 tokens prefill is real compute, and skipping it gives 2.56x.

### Speculative decoding

Qwen3-1.7B target, Qwen3-0.6B draft, 200 tokens.

| draft | tokens drafted | acceptance | vs. target alone |
|---|---|---|---|
| eager | 3 | 58% | 0.62x |
| eager | 5 | 44% | 0.49x |
| CUDA graph | 3 | 58% | 1.74x |
| CUDA graph | 5 | 44% | 1.61x |

Both models have 28 layers, and eager decoding is launch-bound, so the eager draft runs almost as slowly per token as the target and speculation loses. Running the draft as a CUDA graph makes drafting cheap enough to win. Output always matches the target alone exactly.

### Sampling

Each slot has its own temperature and top-p, and sampling runs on the whole batch at once, so greedy and sampled requests can share a batch. Sampling happens outside the CUDA graph, so the graph only produces logits. Speculative decoding stays greedy, since sampling there needs rejection sampling to keep the output distribution exact.

## GPT-2 version


Details behind the numbers in the README. All results are on a T4 with PyTorch 2.11, fp32 unless noted, greedy decoding, medians of repeated runs.

## Single-sequence decode

GPT-2 small, 200 new tokens.

| engine | tok/s | vs. naive |
|---|---|---|
| naive | 88 | 1.0x |
| KV cache | 135 | 1.5x |
| KV cache + CUDA graph | 317 | 3.6x |
| + torch.compile | 404 | 4.6x |
| + INT8 | 629 | 7.2x |

The KV cache only gave 1.5x here. GPT-2 small is so small that most of a decode step is Python and kernel launch overhead, not math, so recomputing a short sequence is cheap. At 500 tokens the gap was 3.1x.

That overhead is what CUDA graphs remove: I record the whole decode step once and replay it with one launch. Graphs need fixed shapes, so attention always runs over the full preallocated cache with a mask, and there can't be any CPU syncs inside the step. Just removing one sync per step (`int(lens.max())`) made decode 8-16% faster, even though attention was now doing extra work.

INT8 was slower than fp32 until I compiled it. I moved the scale to after the matmul and let torch.compile fuse the int8-to-float conversion into the matmul, so the GPU reads 1 byte per weight instead of 4.

## Batched decode

Total tok/s, 200 new tokens per sequence.

| batch | dynamic | CUDA graph | compiled | compiled INT8 |
|---|---|---|---|---|
| 1 | 112 | 314 | 401 | 628 |
| 8 | 733 | 1119 | 1258 | 999 |
| 16 | 1550 | 1880 | 2044 | 1651 |

Since decode is mostly overhead, a batch-16 step costs barely more than a batch-1 step. Total throughput goes up 13.8x in dynamic mode.

INT8 loses at batch 8 and 16. Each weight read now serves 16 tokens, so memory traffic matters less. The profiler also shows the batch-16 matmuls going to cuBLAS, so I think the conversion isn't fused anymore at that size. So INT8 helps at small batch sizes, not large ones.

The slot cache is about 17% slower than the plain KV cache at batch 1 (112 vs. 135), because of the per-slot indexing and masking.

## Continuous batching

32 requests, each wanting 10-200 tokens (seeded), 8 slots. Useful tok/s.

| scheduler | tok/s | vs. static batching |
|---|---|---|
| static batching | 438 | 1.00x |
| continuous | 617 | 1.41x |
| continuous + CUDA graph | 924 | 2.11x |
| continuous + compiled | 1031 | 2.35x |

With static batching, every batch runs until its longest request finishes, and finished slots just sit there. That took 681 decode steps at 61% slot utilization. Continuous batching refills a slot as soon as its request is done: 479 steps at 86%, where the minimum possible is 413. The speedup (1.41x) is almost exactly the step reduction (1.42x), so the scheduler itself adds very little. Most of the leftover gap is at the end of the run, when there's nothing left in the queue to fill freed slots. A real server with requests arriving continuously wouldn't hit that as much.

## Speculative decoding

GPT-2 large target, GPT-2 small draft, 3 guesses per round, 4 prompts x 100 tokens.

| draft | tok/s | speedup | tokens per round |
|---|---|---|---|
| none (target alone) | 45 | 1.00x | |
| KV cache draft | 62 | 1.38x | 3.22 |
| CUDA graph draft | 104 | 2.33x | 3.22 |
| compiled draft | 113 | 2.53x | 3.22 |

From a sweep with the KV cache draft (best guess count):

| target | layers (draft : target) | draft/target step cost | speedup |
|---|---|---|---|
| GPT-2 medium | 12 : 24 | 0.49 | 1.17x |
| GPT-2 large | 12 : 36 | 0.33 | 1.40x |

I originally predicted about 1.8x for GPT-2 large, assuming step cost scales with parameter count. It actually matched the layer ratio for both targets, on my Mac and on the T4, because decode was overhead-bound. Turning the draft into a CUDA graph dropped its relative cost to about 0.13, which took the speedup from 1.38x to 2.33x.

The speedup formula from Leviathan et al. (2023) also undershot when I used accepted/proposed (76%) as the acceptance rate: it predicts 2.78 tokens per round and I measured 3.22. That metric counts every guess after a rejection as rejected, which makes acceptance look lower than it is. Measuring tokens per round directly fixed the mismatch.

Acceptance falls off fast with longer drafts (86% at 1 guess, 54% at 8), so around 3 guesses was best. One caveat: GPT-2 small repeats itself a lot, which makes it easy to predict. On more varied text acceptance would probably be lower.

## INT8 quantization

GPT-2 small, perplexity on the first 20,480 tokens of WikiText-2.

| weights | memory | perplexity |
|---|---|---|
| fp32 | 498 MB | 31.03 |
| INT8, per tensor | 243 MB | 35.40 |
| INT8, per channel | 243 MB | 31.02 |

Same memory, very different quality. With one scale for the whole matrix, a few outlier weights set the scale and every normal weight gets squeezed into a handful of integer values (14% worse perplexity). One scale per output row keeps each outlier in its own row, and perplexity doesn't change. Memory only halves because the embeddings (about 39M parameters) stay in fp32. The quantized linear layers are 4x smaller.

## Comparison with HuggingFace

Same T4, fp32, greedy, exactly 200 new tokens.

| engine | batch 1 (tok/s) | batch 16 (tok/s) |
|---|---|---|
| HF `generate()`, default | 104 | 1373 |
| HF `generate()`, static cache + compile | 258 | 2045 |
| this engine, compiled | 404 | 2044 |

At batch 1 my whole decode step is a single graph replay, while HF still runs a Python loop around each step. At batch 16 both spend most of the step in the same cuBLAS matmuls, so they end up the same. I left INT8 out of this table since HF is running fp32.

## A bug worth writing down

My first torch.compile version ran batch-16 decode at 358 tok/s, compared to about 2,000 without compiling.

The first suspect was recompilation during the timed runs, but `TORCH_LOGS="recompiles"` showed recompiles only happened during warm-up. Turning off automatic dynamic shapes and turning on coordinate descent tuning helped at batch 1, but batch 16 was still 5.6x slower.

Profiling (`bench/profile_decode.py`) showed three copy kernels taking 251 ms of the 362 ms over 10 steps. My K and V caches were each one big tensor for all 12 layers (302 MB at batch 16), and the compiler couldn't update the per-layer views in place, so it copied the whole cache several times every step. Splitting K and V into one tensor per layer got rid of the copies. GPU time dropped from 362 to 91 ms per 10 steps, compiled decode became faster than the plain CUDA graph at every batch size, and INT8 ended up 1.43x faster than fp32 at batch 1.

## Limitations

- GPT-2 only, fp32 only (except the INT8 weights), single GPU.
- Greedy decoding only. Speculative decoding uses the greedy acceptance rule, not rejection sampling.
- Prompts are prefilled one at a time, so a new request briefly stalls the other slots.
- Static-shape decode attends over the whole preallocated cache, which wastes work on short sequences.
- In speculative decoding only the draft runs as a graph, not the target's verification pass.
- The benchmark prompts are a small fixed set.

## Next

- fp16, to use the T4's tensor cores, and then a fair comparison with vLLM
- Batched or chunked prefill
- A paged KV cache so each slot doesn't reserve `max_len`
- Graphing the target's verification pass
- Sampling, with rejection sampling for speculative decoding
