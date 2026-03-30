---
title: "KV Cache is the New Memory Wall: 57x TTFT Reduction at 16k Context"
date: 2026-03-19T15:00:00+09:00
draft: false
tags: ["llm", "inference", "kv-cache", "vllm", "benchmark"]
series: ["InferBench"]
summary: "Without prefix caching, doubling context length more than doubles your time-to-first-token. At 16k tokens, prefill consumes 60% of total latency — 6.3 seconds of waiting. Prefix caching cuts that to 111ms. Here's the data."
ShowToc: true
TocOpen: true
---

In my [previous post](/posts/inferbench-awq-vs-nvfp4/), I benchmarked AWQ vs NVFP4 quantization and showed that prefix caching reduced TTFT by 96.5% on a 7.5k token RAG context. But that experiment used a single context length.

The natural follow-up question: **how does this scale?** If your context grows from 2k to 16k tokens, how much worse does it get? And does prefix caching keep up?

I ran Experiment 3 of [InferBench](https://github.com/KR-LSB/inferbench) to answer this. The results reveal why NVIDIA calls KV cache "the new memory wall."

---

## The Experiment

I swept four context lengths — 2k, 4k, 8k, and 16k tokens — on Qwen3-8B-AWQ running on an RTX 5070 Ti (16GB) via vLLM 0.13.0. For each context length, I sent 5 measurement requests (plus 1 warmup) simulating a RAG workload: a long technical document as context, with different questions appended each time.

I ran this twice: once with prefix caching enabled (default), once with `--no-enable-prefix-caching`.

**Key parameters:**
- Model: Qwen3-8B-AWQ (awq_marlin kernel)
- Output: 256 tokens max per request
- Concurrency: 1 (to isolate context length effects)
- GPU memory utilization: 90%

---

## The Results

### Without Prefix Caching: The Cost Curve

| Context | TTFT (mean) | TPS | Prefill % | Total Latency |
|---|---|---|---|---|
| ~2k | 505 ms | 74.4 | 12.7% | 4,148 ms |
| ~4k | 1,200 ms | 65.3 | 23.6% | 5,119 ms |
| ~8k | 2,547 ms | 62.7 | 38.3% | 6,668 ms |
| ~16k | 6,306 ms | 61.0 | 59.9% | 10,517 ms |

The pattern is clear and punishing:

**TTFT scales super-linearly with context length.** Going from 2k to 16k (8x more tokens) increased TTFT by 12.5x (505ms → 6,306ms). This is worse than linear because longer sequences require more KV cache memory, increasing memory pressure and potentially triggering scheduling delays.

**Prefill dominates at long contexts.** At 2k tokens, prefill is only 12.7% of total request time. At 16k, it's 59.9% — the user spends more time waiting for the model to *read* than to *write*. This is the core insight behind NVIDIA's disaggregated inference: prefill and decode have fundamentally different resource profiles, and lumping them together on one GPU wastes capacity.

**Decode TPS also degrades, but less dramatically.** TPS dropped from 74.4 to 61.0 across the range — an 18% decline. This happens because longer KV caches consume more memory bandwidth during each decode step, even though the computation per token stays the same.

### With Prefix Caching: The Fix

| Context | TTFT (mean) | TPS | Prefill % | Total Latency |
|---|---|---|---|---|
| ~2k | 62 ms | 95.6 | 2.2% | 2,751 ms |
| ~4k | 72 ms | 94.7 | 2.7% | 2,725 ms |
| ~8k | 78 ms | 85.5 | 2.5% | 3,084 ms |
| ~16k | 111 ms | 69.1 | 3.2% | 3,553 ms |

With caching, the picture transforms completely. TTFT stays nearly flat — 62ms to 111ms across an 8x context increase. The cache absorbs the entire prefill cost after the first request, turning a 6.3-second wait into a 111ms blip.

### The Comparison: Cache ON vs OFF

| Context | Cache OFF TTFT | Cache ON TTFT | **Speedup** | Cache OFF Prefill % | Cache ON Prefill % |
|---|---|---|---|---|---|
| ~2k | 505 ms | 62 ms | **8x** | 12.7% | 2.2% |
| ~4k | 1,200 ms | 72 ms | **17x** | 23.6% | 2.7% |
| ~8k | 2,547 ms | 78 ms | **33x** | 38.3% | 2.5% |
| ~16k | 6,306 ms | 111 ms | **57x** | 59.9% | 3.2% |

The speedup from prefix caching doesn't just increase with context length — it *accelerates*. At 2k, caching gives you 8x. At 16k, it's 57x. This is because caching eliminates a cost that itself grows super-linearly.

---

## Three Insights

### 1. KV Cache is the real cost center, not model weights

With AWQ quantization, Qwen3-8B uses only 5.7 GiB of VRAM for model weights. That leaves ~7.2 GiB for KV cache on a 16GB GPU. At 16k context, a single request's KV cache is substantial — and without caching, every request pays the full prefill cost from scratch.

The economic argument is simple: if you're running a RAG service and your users ask multiple questions about the same document, you're burning GPU compute re-processing the same context over and over. Prefix caching turns that from O(n) cost per query to O(1) amortized.

### 2. The "context length tax" hits decode too

Even with caching (which eliminates prefill overhead), TPS dropped from 95.6 at 2k to 69.1 at 16k — a 28% decline. This is purely from the larger KV cache consuming more memory bandwidth during each attention computation in the decode phase.

This has direct implications for pricing: a 16k-context response costs ~38% more compute time than a 2k-context response, even when prefill is cached. This is why cloud providers charge per-token differently based on context length, and why Jensen Huang's "token tiers" make economic sense.

### 3. The warmup cost reveals true prefill expense

The warmup requests (first cold request at each context length) show what prefill actually costs without any caching:

| Context | Cold Prefill (warmup TTFT) |
|---|---|
| ~2k | 607 ms |
| ~4k | 1,245 ms |
| ~8k | 2,599 ms |
| ~16k | 5,911 ms |

Note that the 2k warmup in the Cache ON run showed 26,650ms — this was the very first request after server startup, which includes model initialization overhead. The Cache OFF warmup at 607ms is a more accurate measure of pure 2k prefill cost.

---

## What This Means for Production RAG Systems

If you're building a RAG system, these numbers should change how you think about architecture:

**1. Enable prefix caching. Always.** There's no downside — decode performance is identical (or slightly better due to less memory pressure). The upside at long contexts is a 57x TTFT improvement.

**2. Batch your RAG queries by document.** If five users ask questions about the same document, serve them sequentially on the same vLLM instance. The first request pays the full prefill; the next four get 57x faster TTFT.

**3. Monitor KV cache utilization, not just GPU utilization.** Your GPU might show 50% compute usage but be completely bottlenecked on KV cache memory. vLLM exposes `gpu_cache_usage_perc` via its metrics endpoint — watch it.

**4. Context length is a cost lever.** Trimming your RAG context from 16k to 8k doesn't just save TTFT — it also improves decode TPS by ~10%. Consider whether your retrieval pipeline is sending more context than the model actually needs.

---

## What arXiv:2601.09527 Didn't Measure

The original RTX 5070 Ti benchmark paper reported throughput at fixed context lengths but never varied context length systematically or tested prefix caching. InferBench Experiment 3 adds:

- Context scaling curve (2k → 4k → 8k → 16k) with precise TTFT measurements
- Prefix caching impact at each context length (8x to 57x speedup)
- Prefill ratio analysis showing prefill grows from 12.7% to 59.9% of total latency
- Decode TPS degradation curve as KV cache grows

---

## What's Next

- **SGLang comparison** — Does SGLang handle long contexts differently than vLLM?
- **EXAONE-Deep-7.8B** — Korean-language model on the same context sweep
- **Multi-concurrency × context length** — How does c=8 interact with 16k context?
- **Prometheus + Grafana** — Real-time KV cache utilization dashboards

All code and raw results: **[github.com/KR-LSB/inferbench](https://github.com/KR-LSB/inferbench)**

---

## Hardware & Software

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5070 Ti 16GB GDDR7 (SM_120, Blackwell) |
| CPU | AMD Ryzen 9 9900X (16C/32T) |
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| Driver | 595.79 (CUDA 13.2) |
| vLLM | 0.13.0 (Docker: vllm/vllm-openai:latest) |
| Model | Qwen/Qwen3-8B-AWQ (awq_marlin kernel) |

---

*This is Part 2 of the [InferBench series](/series/inferbench/). Part 1: [AWQ vs NVFP4 and Why Prefix Caching Changes Everything](/posts/inferbench-awq-vs-nvfp4/).*

*SeungByeong is a software engineer focused on LLM inference optimization. [GitHub](https://github.com/KR-LSB) · [InferBench](https://github.com/KR-LSB/inferbench)*
