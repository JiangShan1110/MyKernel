import pytest
import torch

from test_framework.test_abc import TestAbc
from test_framework.utils import load_cutlass_extension


def gemm_golden(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> None:
    c.copy_(torch.matmul(a, b.T))


def gemm_fp16_16_8_8_cuda(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor | None = None,
    **kwargs,
) -> None:
    ext = load_cutlass_extension(
        "gemm_fp16_16_8_8_cuda", "kernel/gemm/gemm_fp16_16_8_8.cu", dump_file=True
    )
    ext.gemm_fp16_16_8_8(a, b, c)


class TestGemmCutlass(TestAbc):
    @pytest.mark.parametrize("shape", [(128, 128, 32)])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_gemm_f16(self, shape, dtype):
        m, n, k = shape
        a = self.get_tensor((m, k), dtype, data=torch.arange(m * k, dtype=dtype) % 8)
        b = self.get_tensor((n, k), dtype, data=torch.arange(k * n, dtype=dtype) % 8)
        c = torch.zeros((m, n), device=a.device, dtype=dtype)

        func_args = {}
        self.invoke(
            [a, b],
            [c],
            func_args=func_args,
            kernel_func=gemm_fp16_16_8_8_cuda,
            golden_func=gemm_golden,
        )
