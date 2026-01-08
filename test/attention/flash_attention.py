import torch


def strandard_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    **attrs,
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
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    output.copy_(torch.matmul(attn_weights, value).to(output.dtype))


def flash_attention_v1(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    **attrs,
) -> None:
    assert (
        query.dim() == 2 and key.dim() == 2 and value.dim() == 2 and output.dim() == 2
    )
    assert query.size(0) == key.size(0) == value.size(0) == output.size(0)
    assert query.size(1) == key.size(1) == value.size(1) == output.size(1)

    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)

    TILE_M = 128
    TILE_N_K = 128
    N = query.size(0)
    D = query.size(1)

    output.fill_(0)
    tmp_max = torch.zeros((N, 1), dtype=torch.float32)
    tmp_max.fill_(-torch.inf)
    tmp_sum = torch.zeros_like(tmp_max)

    for start_n_k in range(0, N, TILE_N_K):
        tile_k = key[start_n_k : start_n_k + TILE_N_K, :]
        tile_v = value[start_n_k : start_n_k + TILE_N_K, :]

        for start_m in range(0, N, TILE_M):
            tile_q = query[start_m : start_m + TILE_M]

            scores = (
                tile_q @ tile_k.T / torch.sqrt(torch.tensor(D, dtype=torch.float32))
            )

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
            output[start_m : start_m + TILE_M, :] = cur_output


q = torch.rand((1024, 1024), dtype=torch.float32)
k = torch.rand((1024, 1024), dtype=torch.float32)
v = torch.rand((1024, 1024), dtype=torch.float32)

out1 = torch.zeros_like(q)
out2 = torch.zeros_like(q)
strandard_attention(q, k, v, out1)
flash_attention_v1(q, k, v, out2)

if torch.allclose(out1, out2, rtol=1e-5, atol=1e-8):
    print("Outputs are close!")
else:
    print("Outputs differ!")
