---
title: "Benchmarking LLM Inference on RTX 5070 Ti: AWQ vs NVFP4 and Why Prefix Caching Changes Everything"
date: 2026-03-19T12:00:00+09:00
draft: false
tags: ["llm", "inference", "nvidia", "vllm", "quantization", "benchmark"]
series: ["InferBench"]
summary: "I benchmarked two quantization formats and prefix caching on NVIDIA's Blackwell consumer GPU. AWQ beat NVFP4 by 56% in throughput, and prefix caching reduced time-to-first-token by 96.5% on a 7.5k token RAG context."
cover:
  image: ""
  alt: "LLM Inference Benchmark Results"
  hidden: false
ShowToc: true
TocOpen: true
---

NVIDIA's GTC 2026 keynote made one thing clear: inference is where the money is now. Jensen Huang introduced the concept of token tiers — Free, High, Premium, Ultra — and unveiled a disaggregated inference architecture that splits prefill and decode across different hardware. The Vera Rubin platform promises 10x inference performance-per-watt over Blackwell.

But here's the thing: the same Blackwell architecture sitting in NVIDIA's data center racks also lives in a $750 consumer GPU — the RTX 5070 Ti. And while a recent paper (arXiv:2601.09527) benchmarked LLM inference on this GPU, it left several critical questions unanswered:

- How do different quantization kernels *actually* perform under load?
- What happens when you separate prefill from decode latency?
- Does prefix caching really matter for RAG workloads?

I built [InferBench](https://github.com/KR-LSB/inferbench) to answer these questions. The results surprised me.

---

## The Setup: Getting vLLM Running on Blackwell SM_120

Before any benchmarks, I had to fight the toolchain. The RTX 5070 Ti uses NVIDIA's SM_120 compute capability (Blackwell consumer), and the software ecosystem is still catching up.

**Problem 1: CUDA version mismatch.** The latest vLLM (0.13.0) requires CUDA 12.9+, but my initial driver (572.83) only supported CUDA 12.8. Solution: upgrade to driver 595.79, which brought CUDA 13.2 support.

**Problem 2: NVFP4 doesn't mean what you think.** The project design doc said `--quantization nvfp4`. That flag doesn't exist in vLLM 0.13.0. The actual supported quantization names are `petit_nvfp4`, `modelopt_fp4`, and others — none of which worked cleanly with the base Qwen3-8B weights. The solution was to use a pre-quantized model: `RedHatAI/Qwen3-8B-NVFP4`, which vLLM auto-detects as `compressed-tensors` and routes through the `flashinfer-cutlass` kernel.

**The working Docker commands:**

```bash
# AWQ model (awq_marlin kernel)
docker run --gpus all --ipc=host -p 8000:8000 \
  -v inferbench-hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B-AWQ \
  --gpu-memory-utilization 0.90 --dtype auto

# NVFP4 model (flashinfer-cutlass kernel)
docker run --gpus all --ipc=host -p 8000:8000 \
  -v inferbench-hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model RedHatAI/Qwen3-8B-NVFP4 \
  --gpu-memory-utilization 0.90 --dtype auto
```

If you're running a Blackwell consumer GPU, save yourself a day of debugging: start with pre-quantized models and let vLLM auto-detect the format.

---

## Experiment 1: The Quantization Ladder — AWQ vs NVFP4

### Hypothesis

NVFP4 uses Blackwell's native FP4 tensor cores (SM_120), so it should outperform AWQ (W4A16) in throughput while using less memory.

### What Actually Happened

At concurrency 1, the two formats are nearly identical:

| Metric | AWQ | NVFP4 |
|---|---|---|
| TPS (mean) | 97.8 | 98.4 |
| TTFT P50 | 37 ms | 42 ms |
| ITL P50 | 10.3 ms | 10.2 ms |
| VRAM (model) | 5.7 GiB | ~4.5 GiB |

No real difference. At a single concurrent request, decode is memory-bandwidth-bound regardless of quantization format. The GPU is waiting for data, not compute.

But scale up concurrency, and the story gets interesting:

**AWQ (awq_marlin kernel):**

| Concurrency | TTFT P50 | Per-request TPS | Aggregate TPS | ITL P50 |
|---|---|---|---|---|
| 1 | 33 ms | 122.0 | 119.9 | 8.2 ms |
| 8 | 54 ms | 113.5 | 885.5 | 8.8 ms |
| 16 | 128 ms | 104.7 | **1,587** | 9.6 ms |
| 32 | 126 ms | 103.6 | 1,576 | 9.7 ms |

**NVFP4 (flashinfer-cutlass kernel):**

| Concurrency | TTFT P50 | Per-request TPS | Aggregate TPS | ITL P50 |
|---|---|---|---|---|
| 1 | 41 ms | 99.0 | 97.4 | 10.1 ms |
| 8 | 57 ms | 66.6 | 524.2 | 15.1 ms |
| 16 | 72 ms | 64.7 | **1,015** | 15.5 ms |
| 32 | 70 ms | 64.7 | 1,016 | 15.5 ms |

### Three Findings That Matter

**1. AWQ crushed NVFP4 in throughput — by 56%.** At peak concurrency (c=16), AWQ delivered 1,587 aggregate tokens/second versus NVFP4's 1,015. The `awq_marlin` kernel is far more mature and optimized than `flashinfer-cutlass` on the current vLLM stack. This is not a hardware limitation — it's a software maturity gap.

**2. NVFP4 won on latency at high concurrency.** At c=16, NVFP4 achieved 72ms TTFT versus AWQ's 128ms — 44% faster. Why? NVFP4's smaller model footprint (4.5 GiB vs 5.7 GiB) leaves more VRAM for KV cache, which means less queuing under load.

**3. The GPU saturates at c=16.** Both formats hit a throughput ceiling between c=16 and c=32 — aggregate TPS flatlined. On 16GB of VRAM with a quantized 8B model, 16 concurrent requests is approximately where you max out the available KV cache space.

### The Takeaway

**Quantization format ≠ guaranteed speedup. Kernel maturity matters more than bit width on current software stacks.**

If you're deploying today, AWQ with the `awq_marlin` kernel is the pragmatic choice on vLLM 0.13.0. NVFP4 has theoretical advantages (native Blackwell FP4 hardware), but the kernel implementation hasn't caught up yet. This will likely change as vLLM optimizes its NVFP4 path — and when it does, the TTFT advantage from smaller memory footprint will compound with better decode throughput.

---

## Experiment 2: Prefill/Decode Disaggregation and Prefix Caching

This is the experiment that arXiv:2601.09527 never ran.

At GTC 2026, NVIDIA's disaggregated inference architecture separates prefill (input processing) from decode (token generation) across different hardware — Rubin GPUs handle prefill, Groq LPUs handle decode. The rationale is that these two phases have fundamentally different compute profiles.

I wanted to measure this on a single consumer GPU: how much of total latency comes from prefill vs decode? And can prefix caching eliminate the prefill bottleneck for RAG workloads?

### Setup

- **Model:** Qwen3-8B-AWQ on vLLM 0.13.0
- **Context:** A 5,800-word technical document (~7,593 tokens)
- **Task:** Answer 8 different questions about the same document
- **Rounds:** 2 (Round 1 = cold KV cache, Round 2 = warm cache)
- **Max output:** 256 tokens per response

I ran this twice: once with vLLM's prefix caching enabled (default), once with `--no-enable-prefix-caching`.

### The Results

**With Prefix Caching ON:**

| Metric | Round 1 (cold) | Round 2 (warm) | Overall |
|---|---|---|---|
| TTFT P50 | 56 ms | 65 ms | 66 ms |
| Decode P50 | 3,480 ms | 3,477 ms | 3,478 ms |
| TPS mean | 73.6 | 73.6 | 73.6 |
| Prefill % of total | 2.0% | 1.8% | **1.9%** |

Note: A warmup request showed TTFT of 1,731ms — the true cost of cold prefill on 7.5k tokens. After that initial cache population, all subsequent requests hit the cached prefix.

**With Prefix Caching OFF:**

| Metric | Round 1 | Round 2 | Overall |
|---|---|---|---|
| TTFT P50 | 1,758 ms | 3,153 ms | 1,787 ms |
| Decode P50 | 3,241 ms | 4,394 ms | 3,277 ms |
| TPS mean | 79.5 | 61.7 | 70.6 |
| Prefill % of total | 36.9% | 39.6% | **38.2%** |

Something unexpected happened in the cache-off run: **Round 2 was slower than Round 1.** TTFT jumped from 1,914ms to 2,832ms, and decode TPS dropped from 79.5 to 61.7. The likely cause is KV cache memory pressure — without eviction, accumulated KV entries from Round 1 squeezed the available memory for Round 2, causing both prefill and decode to degrade.

### The Comparison

| Metric | Cache ON | Cache OFF | Improvement |
|---|---|---|---|
| TTFT P50 | 66 ms | 1,787 ms | **27x faster** |
| TTFT mean | 67 ms | 2,373 ms | **35x faster** |
| Prefill % of total | 1.9% | 38.2% | Prefill nearly eliminated |
| Decode TPS | 73.6 | 70.6 | ~Same |
| TTFT reduction | — | — | **96.5%** |

### What This Means

**1. Prefix caching reduced TTFT by 96.5%** — from nearly 2 seconds down to 66 milliseconds on a 7.5k token RAG context. For any RAG system that reuses the same document context across multiple queries (which is... most RAG systems), this is transformative.

**2. Without caching, prefill consumed 38% of total request time.** My original hypothesis predicted 80%+, but that didn't account for Qwen3-8B's `<think>` reasoning behavior. The model generates substantial internal reasoning tokens before the actual answer, which inflates decode time. On a non-reasoning model, the prefill ratio would likely be much higher.

**3. Decode throughput is unaffected by prefix caching** (73.6 vs 70.6 TPS). This confirms that prefix caching is a pure prefill optimization — it doesn't help or hurt the token generation phase.

**4. Without caching, performance degrades over time.** The Round 2 degradation in the cache-off experiment is a practical warning: if your serving system doesn't manage KV cache memory properly, latency will get worse as you serve more requests, not better.

---

## What arXiv:2601.09527 Missed

The original paper benchmarked RTX 5070 Ti inference and reported impressive numbers: 442–1,044 TPS at concurrency 8–64 with NVFP4. But it measured only end-to-end latency. InferBench adds:

| Dimension | arXiv:2601.09527 | InferBench |
|---|---|---|
| Prefill/Decode separation | E2E only | ✅ Streaming-based separation |
| Prefix caching impact | Not tested | ✅ 96.5% TTFT reduction measured |
| Kernel comparison | NVFP4 only | ✅ AWQ vs NVFP4 kernel battle |
| Memory pressure effects | Not observed | ✅ Degradation without cache management |
| Reproducibility | Docker image | ✅ Docker Compose + GitHub Actions CI |

---

## Key Takeaways

If you're building LLM inference systems — whether on consumer GPUs or in production — here's what these experiments tell you:

1. **Don't assume newer quantization = faster.** Kernel maturity determines real-world performance. AWQ's `awq_marlin` kernel outperformed NVFP4's `flashinfer-cutlass` by 56% in throughput on current vLLM. Check your actual serving stack, not just the paper specs.

2. **Prefix caching is not optional for RAG.** A 27x TTFT reduction (1,787ms → 66ms) is the difference between a usable product and a frustrating one. If your RAG system sends the same context with different questions, enable prefix caching immediately.

3. **Monitor KV cache memory.** Without proper management, performance degrades under sustained load. This is exactly why NVIDIA built disaggregated inference — separating prefill and decode lets you optimize memory management for each phase independently.

4. **Consumer Blackwell GPUs are serious inference hardware.** 1,587 aggregate tokens/second from a $750 GPU is remarkable. For development, testing, and small-scale deployment, you don't need cloud APIs.

---

## What's Next

InferBench is an ongoing project. Coming up:

- **Experiment 3: KV Cache Economics** — How does throughput scale with context length (2k → 4k → 8k → 16k)? What's the cost curve?
- **SGLang comparison** — Same hardware, same models, different engine. Fair fight.
- **EXAONE-Deep-7.8B** — Korean-language model benchmarks on Blackwell. Almost no public data exists for this.
- **Prometheus + Grafana dashboards** — Real-time monitoring for the full serving stack.

All code, configs, and raw results are open source: **[github.com/KR-LSB/inferbench](https://github.com/KR-LSB/inferbench)**

---

## Hardware & Software

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5070 Ti 16GB GDDR7 (SM_120, Blackwell) |
| CPU | AMD Ryzen 9 9900X (16C/32T) |
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| Driver | 595.79 (CUDA 13.2) |
| vLLM | 0.13.0 (Docker: vllm/vllm-openai:latest) |
| Models | Qwen/Qwen3-8B-AWQ, RedHatAI/Qwen3-8B-NVFP4 |

---

*SeungByeong is a software engineer focused on LLM inference optimization. He previously built M.A.R.S., a medical AI system using Encounter-Scoped RAG (SNUBH Datathon, 6th/100 teams, F1 0.92), and contributed to [LangChain](https://github.com/langchain-ai/langchain/pull/34997) for Python 3.14 compatibility. He's currently preparing for roles at frontier AI labs.*

*Discuss this post: [GitHub Issues](https://github.com/KR-LSB/inferbench/issues) · [Source Code](https://github.com/KR-LSB/inferbench)*
