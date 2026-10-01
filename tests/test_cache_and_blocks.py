"""Block pool refcounting and prefix-cache block ownership."""

from __future__ import annotations

import random

import pytest

from llmopt.cache import BlockPool, PrefixCache
from llmopt.engine import Request, SamplingParams
from _helpers import build_engine, greedy_reference


def test_append_and_release_round_trip():
    pool = BlockPool(num_blocks=8, block_size=4)
    pool.append(0, 10)
    assert len(pool.chains[0]) == 3
    pool.release(0)
    assert pool.stats()["num_free"] == 8
    assert pool.chains == {}


def test_partial_tail_block_is_not_reallocated():
    pool = BlockPool(num_blocks=4, block_size=4)
    pool.append(0, 6)
    assert len(pool.chains[0]) == 2
    pool.append(0, 1)
    assert len(pool.chains[0]) == 2  # still fits the partially filled tail


def test_shared_blocks_survive_the_first_release():
    pool = BlockPool(num_blocks=4, block_size=4)
    pool.append(0, 8)
    pool.share_blocks(1, pool.chains[0][:], 8)
    freed = pool.release(0)
    assert freed == []
    assert pool.stats()["num_used"] == 2
    pool.release(1)
    assert pool.stats()["num_free"] == 4


def test_cached_block_is_not_recycled_when_owner_releases():
    """The cache must hold its own reference or a hit returns another sequence's KV."""
    pool = BlockPool(num_blocks=4, block_size=4)
    cache = PrefixCache(pool, block_size=4, max_entries=8)
    blocks = pool.append(0, 8)
    cache.insert([1, 2, 3, 4, 5, 6, 7, 8], blocks)
    pool.mark_shared(blocks)
    pool.release(0)
    # The blocks are still owned by the cache, so they must not be allocatable.
    assert pool.stats()["num_free"] == 2
    assert cache.find([1, 2, 3, 4, 5, 6, 7, 8]) is not None


def test_reset_returns_every_pinned_block():
    pool = BlockPool(num_blocks=16, block_size=4)
    cache = PrefixCache(pool, block_size=4, max_entries=64)
    for i in range(4):
        blocks = pool.append(i, 8)
        cache.insert(list(range(i * 8, i * 8 + 8)), blocks)
        pool.mark_shared(blocks)
        pool.release(i)
    assert pool.stats()["num_free"] < 16
    cache.reset()
    assert pool.stats()["num_free"] == 16
    assert pool.stats()["num_shared"] == 0


def test_demand_eviction_frees_blocks_for_a_new_sequence():
    pool = BlockPool(num_blocks=6, block_size=4)
    cache = PrefixCache(pool, block_size=4, max_entries=64)
    for i in range(3):
        blocks = pool.append(i, 8)
        cache.insert(list(range(i * 8, i * 8 + 8)), blocks)
        pool.mark_shared(blocks)
        pool.release(i)
    assert pool.stats()["num_free"] == 0
    cache.evict_until_free(2)
    assert pool.stats()["num_free"] >= 2


def test_prefix_cache_hit_reproduces_stateless_output():
    engine = build_engine()
    prompt = [(i * 13) % 200 + 1 for i in range(150)]
    first = engine.generate_sync(prompt, SamplingParams(max_tokens=6, temperature=0.0))
    before = engine.scheduler.stats.num_prefix_hits
    second = engine.generate_sync(prompt, SamplingParams(max_tokens=6, temperature=0.0))
    assert engine.scheduler.stats.num_prefix_hits > before
    assert second.token_ids == first.token_ids
    assert second.token_ids == greedy_reference(engine.model, prompt, 6)


def test_prefix_cache_with_chunked_prefill_reproduces_stateless_output():
    engine = build_engine()
    engine.config.scheduler.prefill_chunk_size = 5
    prompt = [(i * 7) % 200 + 1 for i in range(150)]
    out = engine.generate_sync(prompt, SamplingParams(max_tokens=6, temperature=0.0))
    assert out.token_ids == greedy_reference(engine.model, prompt, 6)


def test_engine_fully_reclaims_the_pool_after_a_run():
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
    engine.run(requests)
    assert engine.block_pool.stats()["num_sequences"] == 0
    # Remaining blocks belong to the prefix cache and must be reclaimable.
    engine.reset_prefix_cache()
    assert engine.block_pool.stats()["num_free"] == engine.config.cache.num_gpu_blocks
