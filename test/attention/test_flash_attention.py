import pytest
import torch

from test_framework.test_abc import TestAbc


def standard_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    is_with_causal_mask: bool = False,
    **kwargs,
) -> None:
    """
    A standard implementation of scaled dot-product attention.
    This function computes the attention scores, applies softmax, and computes the weighted sum of values.
    """
    assert (
        query.dim() == 2 and key.dim() == 2 and value.dim() == 2 and output.dim() == 2
    )
    assert query.size(0) == key.size(0) == value.size(0) == output.size(0)
    assert query.size(1) == key.size(1) == value.size(1) == output.size(1)

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )
    if is_with_causal_mask:
        N = scores.size(0)
        causal_mask = torch.triu(
            torch.ones((N, N), device=scores.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))

    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    output.copy_(torch.matmul(attn_weights, value).to(output.dtype))


def flash_attention_v1(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    is_with_causal_mask: bool = False,
    **kwargs,
) -> None:
    assert (
        query.dim() == 2 and key.dim() == 2 and value.dim() == 2 and output.dim() == 2
    )
    assert query.size(0) == key.size(0) == value.size(0) == output.size(0)
    assert query.size(1) == key.size(1) == value.size(1) == output.size(1)

    TILE_M = 32
    TILE_N_K = 32
    N = query.size(0)
    D = query.size(1)
    assert N % TILE_M == 0
    assert N % TILE_N_K == 0

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    output.fill_(0)
    tmp_max = torch.zeros((N, 1), dtype=torch.float32).to(query.device)
    tmp_max.fill_(-torch.inf)
    tmp_sum = torch.zeros_like(tmp_max).to(query.device)

    for start_n_k in range(0, N, TILE_N_K):
        tile_k = key[start_n_k : start_n_k + TILE_N_K, :]
        tile_v = value[start_n_k : start_n_k + TILE_N_K, :]

        for start_m in range(0, N, TILE_M):
            tile_q = query[start_m : start_m + TILE_M]

            scores = (
                tile_q @ tile_k.T / torch.sqrt(torch.tensor(D, dtype=torch.float32))
            )

            if is_with_causal_mask:
                m_indices = torch.arange(
                    start=start_m, end=start_m + TILE_M, device=scores.device
                )[:, None]
                n_indices = torch.arange(
                    start=start_n_k, end=start_n_k + TILE_N_K, device=scores.device
                )[None, :]
                causal_mask = m_indices < n_indices
                scores.masked_fill_(causal_mask, float("-inf"))

            per_max = tmp_max[start_m : start_m + TILE_M, :]
            cur_max = torch.maximum(per_max, torch.max(scores, dim=-1).values[:, None])

            per_sum = tmp_sum[start_m : start_m + TILE_M, :]
            cur_sum = (
                per_sum * torch.exp(per_max - cur_max)
                + torch.sum(torch.exp(scores - cur_max), dim=-1)[:, None]
            )

            per_output = output[start_m : start_m + TILE_M, :]
            cur_output = (
                per_output * per_sum / cur_sum * torch.exp(per_max - cur_max)
                + (torch.exp(scores - cur_max) / cur_sum) @ tile_v
            )

            tmp_max[start_m : start_m + TILE_M, :] = cur_max
            tmp_sum[start_m : start_m + TILE_M, :] = cur_sum
            output[start_m : start_m + TILE_M, :] = cur_output.to(output.dtype)


class TestFlashAttention(TestAbc):
    @pytest.mark.parametrize("shape", [(512, 512), (1024, 1024), (2048, 2048)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    @pytest.mark.parametrize("is_with_causal_mask", [False, True])
    def test_flash_attention_v1_golden(
        self,
        shape,
        dtype,
        is_with_causal_mask,
    ):
        q = self.get_tensor(shape, dtype)
        k = self.get_tensor(shape, dtype)
        v = self.get_tensor(shape, dtype)
        out_flash = torch.zeros_like(q)

        self.invoke(
            [q, k, v],
            [out_flash],
            kwargs={"is_with_causal_mask": is_with_causal_mask},
            kernel_func=flash_attention_v1,
            golden_func=standard_attention,
        )

    @pytest.mark.parametrize("shape", [(128, 128), (2048, 128)])
    @pytest.mark.parametrize("dtype", [torch.float32])
    @pytest.mark.parametrize("is_with_causal_mask", [False])
    def test_flash_attention_v1_triton(
        self,
        shape,
        dtype,
        is_with_causal_mask,
    ):
        q = self.get_tensor(shape, dtype)
        k = self.get_tensor(shape, dtype)
        v = self.get_tensor(shape, dtype)
        out_flash = torch.zeros_like(q)

        from src.attention.flash_attention import flash_attention_v1_triton

        self.invoke(
            [q, k, v],
            [out_flash],
            kwargs={"is_with_causal_mask": is_with_causal_mask},
            kernel_func=flash_attention_v1_triton,
            golden_func=flash_attention_v1,
        )
