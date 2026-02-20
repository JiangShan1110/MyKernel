import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_stages=2, num_warps=4),
    ],
    key=["N", "D"],
)
@triton.jit
def flash_attention_v1_kenerl(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    TILE_M: tl.constexpr,
    TILE_N_K: tl.constexpr,
):
    tl.static_print(f"N={N}")
    tl.static_print(f"D={D}")
    tl.static_print(f"TILE_M={TILE_M}")
    tl.static_print(f"TILE_N_K={TILE_N_K}")

    pid = tl.program_id(0)
    offset = tl.arange(0, D)[None, :]

    tmp_sum = tl.zeros([TILE_M, 1], tl.float32)
    tmp_max = tl.full([TILE_M, 1], -float("inf"), tl.float32)

    for start_n in tl.range(0, N, TILE_N_K):
        tile_n_offset = tl.arange(0, TILE_N_K)[:, None]
        tile_n_offset += start_n
        tile_k = tl.load(k_ptr + tile_n_offset * D + offset)
        tile_v = tl.load(v_ptr + tile_n_offset * D + offset)

        for start_m in tl.range(pid * TILE_M, (pid + 1) * TILE_M, TILE_M):
            tile_m_offset = tl.arange(0, TILE_M)[:, None]
            tile_m_offset += start_m
            tile_q = tl.load(q_ptr + tile_m_offset * D + offset)

            scores = tl.dot(tile_q, tl.trans(tile_k, (1, 0))) / tl.sqrt(1.0 * D).to(
                tl.float32
            )

            cur_max = tl.maximum(tmp_max, tl.max(scores, axis=1, keep_dims=True))

            cur_sum = tmp_sum * tl.exp(tmp_max - cur_max) + tl.sum(
                tl.exp(scores - cur_max), axis=1, keep_dims=True
            )

            per_out = tl.load(out_ptr + tile_m_offset * D + offset)
            cur_out = per_out * tmp_sum / cur_sum * tl.exp(tmp_max - cur_max) + tl.dot(
                tl.exp(scores - cur_max) / cur_sum, tile_v
            )

            tmp_max = cur_max
            tmp_sum = cur_sum
            tl.store(out_ptr + tile_m_offset * D + offset, cur_out)


def flash_attention_v1_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    **kwargs,
):
    assert q.size() == k.size() == v.size()
    assert q.dim() == 2
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype == torch.float32

    N = q.size(0)
    D = q.size(1)
    TILE_M = 32
    TILE_N_K = 32
    assert N % TILE_M == 0
    assert D % TILE_N_K == 0

    grid = ((N + TILE_M - 1) // TILE_M,)
    flash_attention_v1_kenerl[grid](q, k, v, out, N, D, TILE_M, TILE_N_K)


@triton.jit
def flash_attention_v2_kenerl(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    N: tl.constexpr,
    D: tl.constexpr,
    TILE_Q: tl.constexpr,
    TILE_KV: tl.constexpr,
    enable_causal_mask: tl.constexpr = False,
):
    pid = tl.program_id(0)
    offset = tl.arange(0, D)[None, :]

    tile_q_offset = tl.arange(0, TILE_Q)[:, None]
    tile_q_offset += pid * TILE_Q
    mask_q = tile_q_offset < N
    tile_q = tl.load(q_ptr + tile_q_offset * D + offset, mask=mask_q)

    per_sum = tl.zeros([TILE_Q, 1], tl.float32)
    per_max = tl.full([TILE_Q, 1], -float("inf"), tl.float32)
    cur_out = tl.zeros([TILE_Q, D], tl.float32)

    for start_k in tl.range(0, N, TILE_KV):
        tile_kv_offset = tl.arange(0, TILE_KV)[:, None]
        tile_kv_offset += start_k
        mask_kv = tile_kv_offset < N
        tile_k = tl.load(k_ptr + tile_kv_offset * D + offset, mask=mask_kv)
        tile_v = tl.load(v_ptr + tile_kv_offset * D + offset, mask=mask_kv)

        scores = tl.dot(tile_q, tl.trans(tile_k, (1, 0))) / tl.sqrt(1.0 * D).to(
            tl.float32
        )

        if enable_causal_mask:
            mask_causal = tile_q_offset >= tile_kv_offset.T
            scores = tl.where(mask_causal, scores, float("-inf"))

        cur_max = tl.maximum(per_max, tl.max(scores, axis=1, keep_dims=True))
        cur_sum = per_sum * tl.exp(per_max - cur_max) + tl.sum(
            tl.exp(scores - cur_max), axis=1, keep_dims=True
        )

        cur_out = cur_out * per_sum / cur_sum * tl.exp(per_max - cur_max) + tl.dot(
            tl.exp(scores - cur_max) / cur_sum, tile_v
        )

        per_max = cur_max
        per_sum = cur_sum

    tl.store(out_ptr + tile_q_offset * D + offset, cur_out, mask=mask_q)


def flash_attention_v2_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    **kwargs,
):
    assert q.size() == k.size() == v.size()
    assert q.dim() == 2
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype == torch.float32

    N = q.size(0)
    D = q.size(1)
    TILE_Q = 32
    TILE_KV = 32

    grid = ((N + TILE_Q - 1) // TILE_Q,)
    enable_causal_mask = kwargs.get("enable_causal_mask", False)
    flash_attention_v2_kenerl[grid](
        q, k, v, out, N, D, TILE_Q, TILE_KV, enable_causal_mask
    )
