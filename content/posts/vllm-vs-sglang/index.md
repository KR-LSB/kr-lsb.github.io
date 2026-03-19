---
title: "vLLM vs SGLang on Blackwell: Which Engine Wins on RTX 5070 Ti?"
date: 2026-03-19T19:00:00+09:00
draft: false
tags: ["llm", "inference", "vllm", "sglang", "benchmark", "nvidia"]
series: ["InferBench"]
summary: "I ran identical benchmarks on vLLM and SGLang using the same model, same GPU, and same scripts. SGLang delivered 6-47% higher throughput — but vLLM's tail latency was 8x more stable. Here's the full comparison."
ShowToc: true
TocOpen: true
---

The first two InferBench posts focused on a single engine: vLLM. I measured [AWQ vs NVFP4 quantization and prefix caching](/posts/inferbench-awq-vs-nvfp4/), then mapped [how KV cache costs scale with context length](/posts/kv-cache-economics/). Those experiments established the baseline.

But every benchmark comment section asks the same question: **what about SGLang?**

SGLang has been gaining momentum fast. It powers inference at xAI, NVIDIA, AMD, LinkedIn, and Cursor, and claims to be deployed on over 400,000 GPUs globally. Its RadixAttention — a radix-tree-based KV cache management system — is architecturally different from vLLM's PagedAttention. The theory says RadixAttention should handle prefix-heavy workloads more efficiently because it matches at the token level, not the block level.

Theory is nice. I wanted data. So I ran the exact same benchmarks on both engines, on the same GPU, with the same model, using the same scripts.

---

## The Setup: Getting SGLang Running on Blackwell SM_120

My first challenge was figuring out which Docker image to use. The SGLang GitHub tracker ([#5338](https://github.com/sgl-project/sglang/issues/5338)) mentioned a `lmsysorg/sglang:blackwell` tag. It no longer exists — `docker pull` returned `not found`.

After digging through Docker Hub tags and GitHub issues, I found the answer: the `blackwell` tag was merged into the mainline. The `latest` image now includes SM_120 support. No special tag needed.

```bash
# SGLang server (port 30000)
docker run --gpus all --ipc=host -p 30000:30000 --shm-size 16g \
  -v inferbench-hf-cache:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B-AWQ \
  --host 0.0.0.0 --port 30000 --mem-fraction-static 0.90

# vLLM server (port 8000)
docker run --gpus all --ipc=host -p 8000:8000 \
  -v inferbench-hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B-AWQ \
  --gpu-memory-utilization 0.90 --dtype auto
```

Both engines loaded `Qwen/Qwen3-8B-AWQ` from the same cached weights. Both used the same 90% GPU memory allocation. Both ran on the same RTX 5070 Ti.

The critical methodological point: **I wrote one benchmark script and ran it against both engines.** Same prompt, same `max_tokens=256`, same concurrency levels, same warmup procedure. The only thing that changed between runs was the port number.

SGLang uses OpenAI-compatible `/v1/chat/completions`, so the API calls are identical. One quirk: Qwen3's thinking mode is controlled differently. In SGLang, you pass `"chat_template_kwargs": {"enable_thinking": false}` in the request body. For this benchmark, I left thinking ON for both engines to maintain consistency with my previous experiments.

---

## Experiment 4A: Concurrency Benchmark

Same prompt ("Explain the concept of KV cache in transformer inference in detail"), `max_tokens=256`, thinking ON, 2 warmup requests, 10 measurement requests per concurrency level.

### Aggregate Throughput

| Concurrency | vLLM | SGLang | Winner | Delta |
|---|---|---|---|---|
| c=1 | 121.5 tok/s | 130.5 tok/s | SGLang | +7% |
| c=8 | 551.7 tok/s | 810.2 tok/s | SGLang | **+47%** |
| c=16 | 968.8 tok/s | 978.4 tok/s | SGLang | +1% |
| c=32 | 969.4 tok/s | 1,027.8 tok/s | SGLang | +6% |

SGLang wins throughput at every concurrency level. The gap is largest at c=8 (47%), where SGLang's scheduler appears to batch more efficiently in the mid-range. At saturation (c=16 and above), the two engines converge because the GPU itself becomes the bottleneck.

### Time to First Token

| Concurrency | vLLM P50 | SGLang P50 | vLLM P95 | SGLang P95 |
|---|---|---|---|---|
| c=1 | **15.4 ms** | 20.0 ms | **16.3 ms** | 23.2 ms |
| c=8 | **37.3 ms** | 37.4 ms | **37.8 ms** | 299.2 ms |
| c=16 | **40.5 ms** | 69.2 ms | **41.2 ms** | 116.0 ms |
| c=32 | 42.2 ms | **39.7 ms** | **42.6 ms** | 40.0 ms |

This is where the story gets nuanced.

At P50 (median), the two engines trade blows. vLLM is faster at low-to-mid concurrency; SGLang edges ahead at c=32.

But look at P95. vLLM's P95 is nearly identical to its P50 at every level — the ratio never exceeds 1.1x. SGLang's P95 at c=8 is 299ms versus a P50 of 37ms. That is an **8x gap** between median and tail latency.

I initially thought c=8 was a flawed measurement. So I re-ran it with 20 requests instead of 10. The P50 settled at 37.4ms (matching vLLM perfectly), but P95 stayed at 299ms. The spike is real — SGLang's scheduler intermittently delays first-token delivery at moderate concurrency.

### Per-Request Decode Speed

| Concurrency | vLLM | SGLang | Delta |
|---|---|---|---|
| c=1 | 122.5 tok/s | 131.9 tok/s | SGLang +8% |
| c=8 | 107.2 tok/s | 128.1 tok/s | SGLang +19% |
| c=16 | 98.4 tok/s | 100.7 tok/s | SGLang +2% |
| c=32 | 98.5 tok/s | 104.5 tok/s | SGLang +6% |

SGLang decodes faster across the board. At c=1 and c=8, the gap is meaningful (8-19%). This is not about caching — it is a difference in decode scheduling or kernel efficiency.

---

## Experiment 4B: Prefix Caching — RadixAttention vs PagedAttention

This is the comparison I was most interested in. SGLang's RadixAttention is designed for prefix-heavy workloads. vLLM's PagedAttention added prefix caching later. Do they actually perform differently?

### Setup

I wrote a ~1,000-word technical document about LLM inference optimization as the system context. Eight different questions were asked about this document across two rounds:
- **Round 1:** Cache is cold (except for one warmup request)
- **Round 2:** Cache should be fully warm

Sequential requests (c=1) to isolate caching behavior from batching effects.

### Cold Prefill

The warmup request reveals how each engine handles the first encounter with a new context:

| Engine | Warmup TTFT |
|---|---|
| vLLM | 27,354 ms |
| SGLang | 416 ms |

Before you conclude that SGLang is 66x faster at cold prefill — vLLM's 27-second warmup includes model initialization overhead on the first request after server startup. This is a known vLLM behavior I have observed in every previous experiment. It is not representative of steady-state cold prefill, which runs closer to 500-600ms based on my Experiment 3 data.

SGLang's server appears to complete model initialization earlier during startup, so its first request reflects a cleaner cold prefill measurement.

### Cached Performance

After the warmup primes the cache, here is how both engines perform across all 16 requests (8 questions × 2 rounds):

| Metric | vLLM | SGLang | Winner |
|---|---|---|---|
| TTFT P50 | **19.8 ms** | 20.7 ms | vLLM (by 1ms) |
| TTFT mean | **20.6 ms** | 20.9 ms | Tie |
| Decode TPS | 115.6 | **127.7** | SGLang (+10%) |
| Prefill ratio | 0.9% | 1.0% | Tie |

### Round-by-Round Breakdown

**vLLM (PagedAttention):**

| Round | TTFT P50 | TPS | Prefill % |
|---|---|---|---|
| Round 1 (cold) | 21.2 ms | 115.7 | 1.0% |
| Round 2 (warm) | 19.2 ms | 115.6 | 0.9% |

**SGLang (RadixAttention):**

| Round | TTFT P50 | TPS | Prefill % |
|---|---|---|---|
| Round 1 (cold) | 21.4 ms | 127.7 | 1.1% |
| Round 2 (warm) | 20.3 ms | 127.6 | 1.0% |

Both engines show a ~2ms improvement from Round 1 to Round 2 as the cache warms up further. Both reduce prefill to approximately 1% of total request time. Both are remarkably stable across all 8 questions within each round.

The only meaningful difference is decode speed: SGLang's 127.7 tok/s versus vLLM's 115.6 tok/s — a consistent 10% gap that matches what I observed in the concurrency benchmark.

### What This Means for the RadixAttention vs PagedAttention Debate

At this context scale (~1,000 words), **RadixAttention and PagedAttention achieve functionally identical cache performance.** Both hit ~20ms cached TTFT, both reduce prefill to ~1% of total time, both show clean Round 1 → Round 2 improvement.

The theoretical advantage of RadixAttention — token-level granularity versus block-level (16-token) matching — does not produce a measurable difference here. This makes sense: with a shared system prompt of ~1,000 words, the prefix is long enough that block alignment is not a significant overhead. The granularity advantage would likely show up more in scenarios with many short, partially-overlapping prefixes (like a multi-turn chat with branching conversation paths).

---

## The Verdict

### When to Choose SGLang

- **Throughput-sensitive workloads.** SGLang delivered 6-47% higher aggregate TPS. If you need to maximize tokens per second per GPU, SGLang wins.
- **Single-user or low-concurrency serving.** At c=1, SGLang's 10% decode speed advantage directly translates to faster responses.
- **Batch processing.** The throughput advantage at mid-concurrency (c=8: +47%) makes SGLang attractive for offline processing pipelines.

### When to Choose vLLM

- **Latency-sensitive production with SLAs.** vLLM's tail latency is dramatically more stable. When your P99 SLA matters, a P95 of 38ms (vLLM) versus 299ms (SGLang) at c=8 is the difference between meeting and missing your target.
- **Ecosystem maturity.** vLLM has more established Prometheus metrics integration, wider community support, and more production deployment references.
- **Predictability over peak performance.** vLLM's TTFT curve is smooth and monotonic. SGLang's has spikes. For capacity planning, predictable behavior is worth more than a few extra percent of throughput.

### When It Does Not Matter

- **Prefix caching for RAG.** Both achieve ~20ms cached TTFT. Pick either.

---

## Methodology Notes

I want to be explicit about what these benchmarks do and do not prove.

**What they show:** Relative performance of vLLM 0.13.0 and SGLang latest on one specific model (Qwen3-8B-AWQ), one specific GPU (RTX 5070 Ti), one specific workload (short-prompt generation with thinking tokens), and one specific date's Docker images (2026-03-19). Both engines update frequently — these results are a snapshot, not a permanent verdict.

**What they do not show:** Performance on larger models, multi-GPU setups, different quantization formats, or different hardware. SGLang's throughput advantage might be larger or smaller on an H100. vLLM's tail latency stability might be worse or better on a different model.

**The fairness guarantee:** Both engines were benchmarked with a single script (`engine_bench.py` for concurrency, `prefix_cache_bench.py` for caching). The only parameter that changed between runs was the port number. The model weights were loaded from the same cached directory. The GPU was cold-started between engine swaps.

**The AWQ kernel question:** vLLM uses the `awq_marlin` kernel for AWQ quantization. SGLang's kernel implementation is different (not explicitly logged). This kernel difference is real and affects throughput. It is part of the engine comparison, not a confound to be removed — because in practice, you choose an engine and get its kernels.

---

## What's Next

InferBench now has four experiments covering quantization, prefix caching, context scaling, and engine comparison. The remaining items:

- **EXAONE-Deep-7.8B benchmarks** — Korean-language model on Blackwell. Almost no public data exists.
- **Context length scaling on SGLang** — Does SGLang's RadixAttention show advantages at 16k+ tokens where prefix overlap patterns get more complex?
- **Prometheus + Grafana** — Real-time monitoring dashboards for both engines.

All code, scripts, and raw JSON results are open source: **[github.com/KR-LSB/inferbench](https://github.com/KR-LSB/inferbench)**

---

## Hardware & Software

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5070 Ti 16GB GDDR7 (SM_120, Blackwell) |
| CPU | AMD Ryzen 9 9900X (16C/32T) |
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| Driver | 595.79 (CUDA 13.2) |
| vLLM | 0.13.0 (`vllm/vllm-openai:latest`) |
| SGLang | latest (`lmsysorg/sglang:latest`) |
| Model | Qwen/Qwen3-8B-AWQ |

---

*This is Part 3 of the [InferBench series](/series/inferbench/). Part 1: [AWQ vs NVFP4 and Prefix Caching](/posts/inferbench-awq-vs-nvfp4/). Part 2: [KV Cache Economics](/posts/kv-cache-economics/).*

*SeungByeong is a software engineer focused on LLM inference optimization. [GitHub](https://github.com/KR-LSB) · [InferBench](https://github.com/KR-LSB/inferbench)*
