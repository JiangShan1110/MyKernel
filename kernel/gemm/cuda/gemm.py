import torch

from test_framework.utils import load_cutlass_extension


def gemm_fp16_16_8_8_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    ext = load_cutlass_extension(
        "gemm_fp16_16_8_8_cuda", "kernel/gemm/cuda/gemm_fp16_16_8_8.cu"
    )
    ext.gemm_fp16_16_8_8(a, b, c)
