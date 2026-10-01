"""The paged engine must produce exactly what a stateless forward pass does."""

from __future__ import annotations

import random

import pytest

from llmopt.engine import Request, SamplingParams
from _helpers import build_engine, greedy_reference


def test_single_request_matches_stateless(engine):
    prompt = list(range(1, 11))
    out = engine.generate_sync(prompt, SamplingParams(max_tokens=6, temperature=0.0))
    assert out.token_ids == greedy_reference(engine.model, prompt, 6)


def test_long_prompt_spanning_many_blocks():
    # The prompt needs a vocabulary larger than the default 256 to hold 300 ids.
    engine = build_engine(vocab_size=320, num_gpu_blocks=64, block_size=8)
    prompt = list(range(1, 300))
    out = engine.generate_sync(prompt, SamplingParams(max_tokens=8, temperature=0.0))
    assert out.token_ids == greedy_reference(engine.model, prompt, 8)


def test_prompt_at_exact_block_boundary():
    engine = build_engine(block_size=16)
    for length in (15, 16, 17, 32, 33):
        prompt = [(i * 7) % 200 + 1 for i in range(length)]
        out = engine.generate_sync(prompt, SamplingParams(max_tokens=4, temperature=0.0))
        assert out.token_ids == greedy_reference(engine.model, prompt, 4), f"len={length}"


def test_chunked_prefill_matches_stateless():
    engine = build_engine()
    engine.config.scheduler.prefill_chunk_size = 7
    engine.config.scheduler.enable_chunked_prefill = True
    prompt = list(range(1, 200))
    out = engine.generate_sync(prompt, SamplingParams(max_tokens=8, temperature=0.0))
    assert out.token_ids == greedy_reference(engine.model, prompt, 8)


def test_blocks_are_released_after_completion(engine):
    engine.generate_sync([1, 2, 3, 4], SamplingParams(max_tokens=4, temperature=0.0))
    assert engine.block_pool.stats()["num_sequences"] == 0


@pytest.mark.parametrize("seed", range(8))
def test_randomized_configurations_match_stateless(seed):
    rng = random.Random(seed)
    engine = build_engine(
        num_gpu_blocks=rng.choice([16, 48, 256]),
        block_size=rng.choice([8, 16, 32]),
        max_num_seqs=rng.choice([1, 2, 8]),
        enable_prefix_caching=bool(rng.getrandbits(1)),
    )
    if rng.getrandbits(1):
        engine.config.scheduler.prefill_chunk_size = rng.choice([3, 9, 64])
    prompt = [rng.randrange(1, 255) for _ in range(rng.randint(1, 200))]
    num_tokens = rng.randint(1, 8)
    out = engine.generate_sync(prompt, SamplingParams(max_tokens=num_tokens, temperature=0.0))
    assert out.token_ids == greedy_reference(engine.model, prompt, num_tokens)


def test_ragged_prompts_run_concurrently_and_all_complete():
    """Mixed prompt lengths in one step used to abort prefill batching."""
    engine = build_engine()
    rng = random.Random(3)
    requests = [
        Request(
            request_id=f"r{i}",
            prompt_token_ids=[rng.randrange(1, 255) for _ in range(rng.randint(5, 300))],
            sampling_params=SamplingParams(
                max_tokens=rng.randint(1, 10), temperature=0.0, ignore_eos=True
            ),
            max_model_len=512,
            arrival_time=0.0,
        )
        for i in range(40)
    ]
    outputs = engine.run(requests)
    assert len(outputs) == 40
    assert all(request.is_done() for request in requests)
    assert engine.block_pool.stats()["num_sequences"] == 0


def test_memory_pressure_still_drains():
    """A pool far smaller than total demand must still complete every request."""
    engine = build_engine(num_gpu_blocks=24)
    rng = random.Random(11)
    requests = [
        Request(
            request_id=f"p{i}",
            prompt_token_ids=[rng.randrange(1, 255) for _ in range(rng.randint(30, 200))],
            sampling_params=SamplingParams(
                max_tokens=rng.randint(2, 12), temperature=0.0, ignore_eos=True
            ),
            max_model_len=512,
            arrival_time=0.0,
        )
        for i in range(16)
    ]
    outputs = engine.run(requests)
    assert len(outputs) == 16
    assert engine.block_pool.stats()["num_sequences"] == 0
