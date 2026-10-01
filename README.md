# Quantized LLM Inference Engine (`llmopt`)

An LLM inference optimization engine in pure PyTorch: weight quantization, a
paged KV cache, continuous batching, prefix caching, and an OpenAI-compatible
HTTP surface — with no compiled extensions required.

> **CPU note.** Quantized weights are stored packed but dequantized to float for
> each matmul, so there is no int4/int8 GEMM kernel. On CPU this saves memory
> rather than time, and quantized layers are somewhat slower than plain
> `nn.Linear`. The throughput wins come from continuous batching, the paged KV
> cache, and prefix caching. See [Performance notes](#performance-notes).

```
pip install -e .          # core
pip install -e ".[dev]"   # + pytest, ruff, mypy
```

## What is implemented

| Area | Module | Notes |
| --- | --- | --- |
| Quantization | `llmopt.quantization` | 2/4/8-bit group-wise RTN and a GPTQ-style path, packed storage, calibration |
| KV cache | `llmopt.cache` | Block table, refcounted block pool, content-addressed prefix cache |
| Attention | `llmopt.attention` | Paged attention with an exact log-sum-exp split-K (flash-decoding) path |
| Engine | `llmopt.engine` | Continuous batching, chunked prefill, preemption, sampling, tokenizer |
| Model | `llmopt.model` | Decoder-only transformer with GQA, RoPE, RMSNorm, cache-aware forward |
| Serving | `llmopt.serving` | OpenAI-compatible endpoints on the standard library, plus a FastAPI app |
| Benchmark | `llmopt.benchmark` | Throughput, TTFT/ITL percentiles, configuration comparison |

## Quick start

```python
from llmopt import LLMEngine, SamplingParams

engine = LLMEngine.from_preset("tiny", max_model_len=1024, block_size=16)
out = engine.generate_sync("Hello", SamplingParams(max_tokens=32, temperature=0.0))
print(out.text, out.finish_reason)
```

`from_preset` builds a randomly initialised model, so nothing is downloaded.
Use `from_pretrained(path)` to serve a checkpoint saved with
`DecoderOnlyModel.save_pretrained`.

## Serving

```bash
llmopt serve --preset tiny --port 8000
# or: python -m llmopt.serving --preset tiny --port 8000
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/v1/models` | Model listing |
| `POST` | `/v1/completions` | Text completion |
| `POST` | `/v1/chat/completions` | Chat completion |
| `GET` | `/metrics` | Prometheus text exposition |
| `GET` | `/info` | Engine and cache configuration |

## Benchmarks

```bash
llmopt bench --num-requests 32 --prompt-tokens 64 --output-tokens 64
llmopt bench --num-requests 32 --shared-prefix 0.8   # prefix-cache workload
```

## How it fits together

A request moves through four stages:

1. **Admission.** The scheduler applies any cached prefix, reserves blocks, and
   plans a prefill chunk.
2. **Prefill.** `ModelRunner` packs the chunks into one batch. The attention
   kernel gathers each sequence's K/V from its physical blocks and applies a
   causal mask, so ragged lengths in a batch are fine.
3. **Decode.** Running sequences advance one token per step, in a single batched
   forward pass, with no per-sequence Python loop.
4. **Release.** Finished requests return their blocks. Blocks that filled up are
   pinned by the prefix cache and reused by later requests with the same prefix.

Two invariants are worth knowing, because they explain the scheduler's shape:

- **Decode is scheduled first and runs alone.** The runner packs prefill ahead of
  decode and drops decode rows when both are present. Admitting new sequences
  before running ones drain would pin every block and leave nothing able to
  finish, so a step is either decode-only or prefill-only.
- **Prefill chunks in one step share a width.** The kernel takes a single
  `q_len` for the batch, so a step batches chunks of equal length and defers the
  rest.

## Correctness

The engine is validated against a stateless reference: a full forward pass per
token with no cache, no block pool, and no scheduler. Any divergence is a bug in
the serving machinery rather than in the model.

```bash
pytest                    # correctness, paged attention, cache refcounts, serving
ruff check . && mypy      # lint and types
```

## Performance notes

Quantization here is a **storage** optimisation, not a compute one.
`QuantLinear.forward` unpacks the codes and dequantizes to `compute_dtype`
before calling `F.linear`, so the matmul itself stays in floating point.

| | Effect |
| --- | --- |
| Memory footprint | Real win. 4-bit weights hold `bits` per weight at rest |
| Single-request latency | No win, and a small loss from dequantization overhead |
| Concurrent throughput | Real win, from batching plus paged allocation and prefix reuse |

If you are serving on CPU, leave the model in `float32` and lean on
`llmopt bench --shared-prefix` to measure the prefix cache. If you need CPU
speed at low bit-width, use a runtime with fused kernels (`llama.cpp`/GGUF,
`torchao`, or `bitsandbytes`) rather than this path.

## License

Apache-2.0. See [LICENSE](LICENSE).
