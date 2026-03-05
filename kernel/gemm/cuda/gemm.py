import torch

from test_framework.utils import load_cutlass_extension


def gemm_f16_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
    *,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a and b must be CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise ValueError("a and b must be float16 tensors")
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError("a.shape[1] must equal b.shape[0]")

    if c is None:
        c = torch.empty((a.shape[0], b.shape[1]), device=a.device, dtype=a.dtype)
    else:
        if not c.is_cuda or c.dtype != torch.float16:
            raise ValueError("c must be a CUDA float16 tensor")
        if c.shape != (a.shape[0], b.shape[1]):
            raise ValueError("c shape must be (a.shape[0], b.shape[1])")
        if not c.is_contiguous():
            raise ValueError("c must be contiguous")

    ext = load_cutlass_extension("gemm_fp16", "kernel/gemm/cuda/gemm.cu")
    ext.gemm_f16(a.contiguous(), b.contiguous(), c, float(alpha), float(beta))
    return c
