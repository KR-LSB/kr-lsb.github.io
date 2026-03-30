---
title: "EXAONE Deep 7.8B on Blackwell: Korean LLM Inference, Benchmarked"
date: 2026-03-30
draft: false
tags: ["LLM", "inference", "EXAONE", "vLLM", "SGLang", "RTX 5070 Ti", "Blackwell", "benchmark"]
series: ["InferBench"]
summary: "First public inference benchmark of LG AI Research's EXAONE-Deep-7.8B on a consumer Blackwell GPU. 2x throughput vs Qwen3-8B at high concurrency, 11x TTFT reduction with prefix caching on Korean RAG, and a head-to-head vLLM vs SGLang comparison."
---

## TL;DR

- **EXAONE-Deep-7.8B-AWQ** delivers **2x aggregate throughput** vs Qwen3-8B-AWQ at c=32 on RTX 5070 Ti (1,934 vs 969 TPS)
- Prefix caching reduces Korean RAG TTFT by **11.3x** (184.9 → 16.4 ms) on vLLM
- SGLang wins on **decode speed** (+10%) and **tail latency at high concurrency**, vLLM wins on **stability at mid-range concurrency**
- Both models hit ~1,940 TPS peak — the GPU is the ceiling, not the model
- Getting EXAONE running on SGLang required patching a `transformers` compatibility issue — details in the troubleshooting section

## Why EXAONE?

[EXAONE-Deep-7.8B](https://huggingface.co/LGAI-EXAONE/EXAONE-Deep-7.8B) is LG AI Research's reasoning-specialized Korean LLM. While there are plenty of Qwen and Llama benchmarks on consumer GPUs, **there is virtually no public data on EXAONE inference performance on Blackwell hardware**. This post fills that gap.

There's also a personal connection: I fine-tuned EXAONE 7.8B for the [M.A.R.S. project](https://github.com/KR-LSB/M.A.R.S.) (medical AI, SNUBH Datathon 6th/100 teams), so benchmarking its inference characteristics is a natural follow-up.

### Test Setup

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5070 Ti 16GB GDDR7 (SM_120, Blackwell) |
| CPU | AMD Ryzen 9 9900X (16C/32T) |
| OS | Windows 11 + WSL2 + Docker |
| vLLM | 0.13.0 (prefix caching ON by default) |
| SGLang | 0.5.9 |
| Model | LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ (awq_marlin kernel, 4.96 GiB VRAM) |
| Baseline | Qwen/Qwen3-8B-AWQ (awq_marlin kernel, 5.7 GiB VRAM) |

Both models use the same `awq_marlin` kernel — this is an apples-to-apples comparison.

---

## Experiment 5A: Concurrency Scaling

The first question: how does EXAONE scale under load compared to Qwen3?

### EXAONE vs Qwen3 on vLLM

| c | Model | Agg TPS | TTFT P50 | TTFT P95 | Decode TPS |
|--:|-------|--------:|---------:|---------:|-----------:|
| 1 | Qwen3-8B | 121.5 | 15.4 ms | 15.8 ms | 121.5 |
| 1 | **EXAONE-7.8B** | **132.0** | **14.6 ms** | 87.0 ms | **133.3** |
| 8 | Qwen3-8B | 551.7 | 37.3 ms | 37.8 ms | 70.0 |
| 8 | **EXAONE-7.8B** | **826.6** | **26.9 ms** | 39.2 ms | **125.3** |
| 16 | Qwen3-8B | 968.8 | 40.5 ms | 42.0 ms | 62.0 |
| 16 | **EXAONE-7.8B** | **1,186.8** | 98.2 ms | 99.4 ms | **118.7** |
| 32 | Qwen3-8B | 969.4 | 42.2 ms | 45.0 ms | 31.0 |
| 32 | **EXAONE-7.8B** | **1,934.1** | 106.1 ms | 117.4 ms | **100.7** |

### Key Finding: EXAONE Doesn't Saturate at c=16

Qwen3 plateaus at c=16 (968 → 969 TPS), but EXAONE keeps scaling to c=32 (1,187 → 1,934 TPS). Why?

**VRAM is the answer.** EXAONE loads at 4.96 GiB vs Qwen3's 5.7 GiB — that's 0.74 GiB more KV cache headroom. The vLLM server reported 68,752 tokens of KV cache capacity and a maximum concurrency of 2.1x for 32K-token requests. More cache space means more concurrent requests can be batched before the scheduler starts queuing.

### EXAONE: vLLM vs SGLang

| c | Engine | Agg TPS | TTFT P50 | TTFT P95 | Decode TPS |
|--:|--------|--------:|---------:|---------:|-----------:|
| 1 | vLLM | 132.0 | **14.6 ms** | 87.0 ms | 133.3 |
| 1 | SGLang | **143.7** | 16.8 ms | 200.2 ms | **145.9** |
| 8 | vLLM | 826.6 | **26.9 ms** | **39.2 ms** | 125.3 |
| 8 | SGLang | **889.9** | 27.7 ms | 288.7 ms | **140.7** |
| 16 | vLLM | 1,186.8 | 98.2 ms | 99.4 ms | 118.7 |
| 16 | SGLang | **1,247.1** | **81.5 ms** | **82.4 ms** | **121.0** |
| 32 | vLLM | 1,934.1 | 106.1 ms | 117.4 ms | **100.7** |
| 32 | SGLang | **1,941.3** | **50.6 ms** | **51.5 ms** | 99.0 |

The pattern mirrors what we saw with Qwen3 in [Blog 3]({{< ref "/posts/vllm-vs-sglang" >}}):

1. **SGLang throughput is consistently 5-8% higher** across all concurrency levels
2. **SGLang decode TPS is 10% faster** (145.9 vs 133.3 at c=1) — this is a consistent engine-level advantage
3. **vLLM is more stable at c=1-8** (TTFT P95: 39ms vs 289ms at c=8)
4. **SGLang wins at high concurrency** — at c=32, TTFT P50 is 50.6ms vs vLLM's 106.1ms

The c=8 P95 spike on SGLang (288.7ms) is the same intermittent pattern we observed with Qwen3. It doesn't appear at c=16 or c=32, suggesting it's a scheduling artifact during the transition from low to medium load.

---

## Experiment 5B: Korean RAG Prefix Caching

RAG workloads reuse the same document context across multiple queries. Prefix caching exploits this by keeping the context's KV cache in memory. For EXAONE — which is designed for Korean language tasks — this is a critical optimization.

### Setup

- **Context:** ~2,000 characters of Korean technical documentation (covering KV cache, quantization, and inference optimization)
- **Questions:** 5 different questions over the same context, 20 requests each
- **EXAONE-specific:** No system prompt (per EXAONE documentation), context embedded in user message

### vLLM: Cache ON vs Cache OFF

| Metric | Cache OFF | Cache ON | Improvement |
|--------|----------:|---------:|------------:|
| **TTFT P50** | 184.9 ms | **16.4 ms** | **11.3x** (91% reduction) |
| TTFT P95 | 186.6 ms | 77.8 ms | 2.4x |
| TTFT Mean | 184.5 ms | 20.6 ms | 9.0x |
| Decode TPS | 130.6 | 129.7 | — (unchanged) |
| Prefill ratio | 8.60% | 1.03% | 8.3x reduction |

The ~2K context produces a consistent 185ms prefill without caching. With caching, the second request onward drops to 15-17ms — that's the cost of just processing the question portion.

### Cache Warmth Pattern

```
Request 1 (cold):     188.8 ms  ← Full context + question prefill
Request 2+ (warm):     16-17 ms ← Only question prefill (context cached)
New question (partial): ~78 ms  ← Context cached, new question suffix prefill
```

The partial cache hit (~78ms for a new question with cached context) shows that vLLM's hash-based prefix matching works at sub-prompt granularity — the shared context prefix is reused even when the question suffix changes.

### vLLM vs SGLang: Prefix Caching Comparison

| Metric | vLLM | SGLang |
|--------|-----:|-------:|
| Cached TTFT P50 | **16.4 ms** | 17.0 ms |
| Cached TTFT P95 | 77.8 ms | **22.2 ms** |
| Decode TPS | 129.7 | **142.7** |
| Prefill ratio | 1.03% | **1.05%** |

Both engines achieve nearly identical cached TTFT (~17ms). SGLang's tighter P95 (22.2 vs 77.8ms) suggests more consistent cache hit behavior. The decode TPS advantage (+10% for SGLang) persists — this is an engine-level constant, independent of caching.

### Cross-Model: EXAONE vs Qwen3 Prefix Caching

| Metric | Qwen3 (vLLM) | EXAONE (vLLM) | EXAONE (SGLang) |
|--------|-------------:|--------------:|----------------:|
| Cached TTFT P50 | 19.8 ms | **16.4 ms** | 17.0 ms |
| Decode TPS | 115.6 | 129.7 | **142.7** |

EXAONE's cached TTFT is 17% faster than Qwen3 on the same engine. Combined with the higher decode speed, EXAONE delivers a noticeably snappier RAG experience.

---

## The Complete Picture

Combining all InferBench data, here's how the four configurations compare:

| Configuration | Peak Agg TPS | Cached RAG TTFT | Decode TPS | VRAM |
|--------------|-------------:|----------------:|-----------:|-----:|
| Qwen3 + vLLM | 969 (c=16) | 19.8 ms | 115.6 | 5.7 GiB |
| Qwen3 + SGLang | 1,028 (c=32) | 20.7 ms | 127.7 | 5.7 GiB |
| EXAONE + vLLM | 1,934 (c=32) | 16.4 ms | 129.7 | 4.96 GiB |
| **EXAONE + SGLang** | **1,941 (c=32)** | **17.0 ms** | **142.7** | **4.96 GiB** |

EXAONE + SGLang is the throughput champion. EXAONE + vLLM has the lowest cached TTFT. All four achieve sub-20ms cached RAG latency.

---

## Troubleshooting: Getting EXAONE on SGLang

This wasn't plug-and-play. Here's the journey for anyone trying to run EXAONE AWQ models on SGLang:

### Problem 1: `ImportError: RopeParameters`

EXAONE's custom `configuration_exaone.py` imports `RopeParameters` from `transformers.modeling_rope_utils`, which only exists in transformers 5.x. SGLang ships with 4.57.1.

**Failed fix:** Upgrading transformers to 5.x breaks SGLang (`sglang 0.5.9 requires transformers==4.57.1`).

### Problem 2: `NoneType | NoneType`

Patching `RopeParameters = None` fails because the class uses `rope_parameters: RopeParameters | None = None` as a type hint. `None | None` is not a valid Python type union.

### Solution: Patch with `typing.Any`

```bash
sed -i 's/from transformers.modeling_rope_utils import RopeParameters/try:\n    from transformers.modeling_rope_utils import RopeParameters\nexcept ImportError:\n    from typing import Any\n    RopeParameters = Any/' "$FILE"
```

`Any | None` is a valid type expression, and `RopeParameters` is only used in type hints — it never affects runtime inference behavior.

### Problem 3: `--trust-remote-code` Removed

SGLang 0.5.9 removed the `--trust-remote-code` flag entirely. Just drop it from the command.

### Final Working Command

```bash
# Patch the cached config file, then launch
find /root/.cache/huggingface/modules -name "configuration_exaone.py" -path "*EXAONE*AWQ*" | \
  while read f; do
    sed -i 's/from transformers.modeling_rope_utils import RopeParameters/try:\n    from transformers.modeling_rope_utils import RopeParameters\nexcept ImportError:\n    from typing import Any\n    RopeParameters = Any/' "$f"
  done

python3 -m sglang.launch_server \
  --model-path LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ \
  --host 0.0.0.0 --port 30000 \
  --mem-fraction-static 0.90
```

For vLLM, the fix is simpler — just upgrade transformers inside the container:

```bash
pip install --upgrade transformers
vllm serve LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ \
  --gpu-memory-utilization 0.90 --dtype auto --trust-remote-code
```

---

## What's Next

This completes InferBench's four original goals:

1. ✅ Prefill vs Decode disaggregated measurement ([Blog 1]({{< ref "/posts/inferbench-awq-vs-nvfp4" >}}))
2. ✅ KV cache optimization / prefix caching ([Blog 2]({{< ref "/posts/kv-cache-economics" >}}))
3. ✅ Engine comparison: vLLM vs SGLang ([Blog 3]({{< ref "/posts/vllm-vs-sglang" >}}))
4. ✅ Korean model (EXAONE) on Blackwell (this post)

The full benchmark suite, scripts, and raw results are available at [github.com/KR-LSB/inferbench](https://github.com/KR-LSB/inferbench).

---

*Benchmarked on RTX 5070 Ti 16GB, Windows 11 + WSL2 + Docker. All results are from EXAONE-Deep-7.8B-AWQ and Qwen3-8B-AWQ using awq_marlin kernels. Prefix caching tests used ~2K Korean technical document context with sequential requests.*
