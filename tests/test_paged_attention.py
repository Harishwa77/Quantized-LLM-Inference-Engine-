"""Paged attention must be numerically identical to dense causal attention."""

from __future__ import annotations

import math

import pytest
import torch

from llmopt.attention import PagedAttention, PagedAttentionConfig
from llmopt.cache import BlockPool, PagedKVCache

SEQ, HEADS, KV_HEADS, HEAD_DIM, BLOCK = 3, 4, 2, 8, 8


def _reference(q, k, v, ctx, q_len, group):
    """Dense per-sequence causal attention."""
    out = torch.empty(q.shape[0], HEADS, q_len, HEAD_DIM)
    for s, length in enumerate(ctx):
        keys = k[s, :length].permute(1, 0, 2).repeat_interleave(group, 0)
        values = v[s, :length].permute(1, 0, 2).repeat_interleave(group, 0)
        scores = (q[s].float() @ keys.float().transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        q_pos = torch.arange(length - q_len, length)[:, None]
        k_pos = torch.arange(length)[None, :]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        out[s] = torch.softmax(scores, -1) @ values.float()
    return out


@pytest.fixture
def paged():
    torch.manual_seed(1)
    pool = BlockPool(num_blocks=64, block_size=BLOCK)
    cache = PagedKVCache(
        num_layers=1,
        num_blocks=64,
        block_size=BLOCK,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float32,
    )
    ctx = [12, 20, 9]  # deliberately ragged, and none a multiple of BLOCK
    table = torch.full((SEQ, 8), -1, dtype=torch.long)
    for s, length in enumerate(ctx):
        table[s, : len(pool.append(s, length))] = torch.tensor(pool.chains[s])
    keys = torch.randn(SEQ, max(ctx), KV_HEADS, HEAD_DIM)
    values = torch.randn(SEQ, max(ctx), KV_HEADS, HEAD_DIM)
    # token_offsets is the start position, so a full prefill starts at zero.
    cache.write(0, table, torch.zeros(SEQ, dtype=torch.long), keys, values)
    return cache, table, keys, values, torch.tensor(ctx)


@pytest.mark.parametrize("split_kv", [False, True])
def test_ragged_batched_prefill_matches_dense(paged, split_kv):
    cache, table, keys, values, ctx = paged
    q_len = 5
    q = torch.randn(SEQ, HEADS, q_len, HEAD_DIM)
    attn = PagedAttention(
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        config=PagedAttentionConfig(block_size=BLOCK, split_kv=split_kv, split_size=4),
    )
    out = attn.forward(q, cache, 0, table, ctx, q_len)
    want = _reference(q, keys, values, ctx.tolist(), q_len, HEADS // KV_HEADS)
    assert torch.allclose(out.float(), want, atol=1e-5)


def test_decode_matches_dense(paged):
    cache, table, keys, values, ctx = paged
    q = torch.randn(SEQ, HEADS, 1, HEAD_DIM)
    attn = PagedAttention(HEADS, KV_HEADS, HEAD_DIM, config=PagedAttentionConfig(block_size=BLOCK))
    out = attn.forward(q, cache, 0, table, ctx, 1)
    want = _reference(q, keys, values, ctx.tolist(), 1, HEADS // KV_HEADS)
    assert torch.allclose(out.float(), want, atol=1e-5)


def test_split_kv_agrees_with_dense(paged):
    cache, table, keys, values, ctx = paged
    q = torch.randn(SEQ, HEADS, 4, HEAD_DIM)
    dense = PagedAttention(
        HEADS, KV_HEADS, HEAD_DIM, config=PagedAttentionConfig(block_size=BLOCK, split_kv=False)
    ).forward(q, cache, 0, table, ctx, 4)
    split = PagedAttention(
        HEADS,
        KV_HEADS,
        HEAD_DIM,
        config=PagedAttentionConfig(block_size=BLOCK, split_kv=True, split_size=3),
    ).forward(q, cache, 0, table, ctx, 4)
    assert torch.allclose(dense, split, atol=1e-5)


def test_cache_round_trips_written_kv(paged):
    cache, table, keys, values, ctx = paged
    got_keys, _, _ = cache.gather(0, table, ctx)
    for s, length in enumerate(ctx.tolist()):
        assert torch.allclose(got_keys[s, :length], keys[s, :length])
