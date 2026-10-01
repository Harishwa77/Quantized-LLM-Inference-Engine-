"""Scheduler bookkeeping, quantization, and the OpenAI-compatible HTTP surface."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest
import torch
from torch import nn

from llmopt.benchmark.runner import run_benchmark
from llmopt.benchmark.workload import WorkloadSpec
from llmopt.engine import Request, SamplingParams
from llmopt.quantization import quantize_model_
from llmopt.quantization.quant_linear import QuantLinear
from llmopt.serving.app import ServerConfig, create_server
from _helpers import build_engine


def test_scheduler_counters_advance():
    """Counters must reflect real work, not stay at zero."""
    engine = build_engine()
    requests = [
        Request(
            request_id=f"r{i}",
            prompt_token_ids=list(range(10, 50)),
            sampling_params=SamplingParams(max_tokens=8, temperature=0.0, ignore_eos=True),
            max_model_len=512,
            arrival_time=0.0,
        )
        for i in range(6)
    ]
    engine.run(requests)
    stats = engine.scheduler.stats
    assert stats.num_scheduled_steps > 0
    assert stats.num_admitted == 6
    assert stats.num_finished == 6
    assert stats.num_prefill_steps > 0
    assert stats.num_decode_steps > 0
    # Six near-identical prompts share a prefix.
    assert stats.num_prefix_hits == 5
    assert stats.total_prefill_tokens > 0


def test_stats_are_exposed_on_the_engine():
    engine = build_engine()
    engine.generate_sync([1, 2, 3], SamplingParams(max_tokens=2, temperature=0.0))
    stats = engine.stats()
    assert {"scheduler", "block_pool", "prefix_cache", "kv_cache_bytes"} <= set(stats)


def test_abort_releases_capacity():
    engine = build_engine()
    engine.generate([1, 2, 3, 4], SamplingParams(max_tokens=64, temperature=0.0), request_id="doomed")
    assert engine.abort("doomed")
    engine.run_until_idle()
    assert engine.block_pool.stats()["num_sequences"] == 0


def test_fully_cached_prompt_is_admitted_and_finishes():
    """A prompt served entirely from the prefix cache must not re-queue forever."""
    engine = build_engine()
    prompt = [(i * 5) % 200 + 1 for i in range(64)]
    engine.generate_sync(prompt, SamplingParams(max_tokens=4, temperature=0.0))
    out = engine.generate_sync(prompt, SamplingParams(max_tokens=4, temperature=0.0))
    assert len(out.token_ids) == 4


def test_quantize_model_replaces_linears_and_shrinks_storage():
    model = build_engine().model
    float_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    quantize_model_(model, bits=4, group_size=64)
    layers = [m for m in model.modules() if isinstance(m, QuantLinear)]
    assert layers, "no linear layers were quantized"
    # lm_head stays in floating point.
    assert any(isinstance(m, nn.Linear) for m in model.modules())
    packed = sum(
        m.qweight.numel() * m.qweight.element_size()
        + m.scales.numel() * m.scales.element_size()
        + m.qzeros.numel() * m.qzeros.element_size()
        for m in layers
    )
    assert packed < float_bytes


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_model_still_runs(bits):
    model = build_engine().model
    quantize_model_(model, bits=bits, group_size=64)
    ids = torch.randint(0, 256, (1, 8))
    out = model(ids, torch.arange(8)[None])
    assert torch.isfinite(out.logits).all()


def test_benchmark_handles_ragged_prompt_lengths():
    engine = build_engine(max_model_len=512, num_gpu_blocks=512)
    spec = WorkloadSpec(
        num_requests=12, prompt_tokens=(16, 200), output_tokens=(4, 12), shared_prefix=0.5, seed=1
    )
    result = run_benchmark(engine, spec, name="ragged", warmup=1)
    # warmup requests are measured separately and excluded.
    assert result.num_requests == spec.num_requests - 1
    metrics = result.to_dict()
    assert metrics["output_tokens"] > 0
    assert metrics["output_throughput_tok_s"] > 0


@pytest.fixture
def server():
    engine = build_engine(max_model_len=256, num_gpu_blocks=256)
    httpd, _ = create_server(engine, ServerConfig(port=0))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.4)
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()


def _call(port, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def test_models_endpoint(server):
    status, body = _call(server, "GET", "/v1/models")
    assert status == 200
    assert json.loads(body)["object"] == "list"


def test_metrics_endpoint(server):
    assert _call(server, "GET", "/metrics")[0] == 200


def test_completions_endpoint(server):
    status, body = _call(
        server, "POST", "/v1/completions",
        {"model": "tiny", "prompt": "hello world", "max_tokens": 5, "temperature": 0.0},
    )
    assert status == 200
    payload = json.loads(body)
    assert payload["object"] == "text_completion"
    assert payload["choices"][0]["finish_reason"] == "length"
    assert len(payload["choices"][0]["token_ids"]) == 5


def test_chat_completions_endpoint(server):
    status, body = _call(
        server, "POST", "/v1/chat/completions",
        {
            "model": "tiny",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 5,
            "temperature": 0.0,
        },
    )
    assert status == 200
    assert json.loads(body)["object"] == "chat.completion"
