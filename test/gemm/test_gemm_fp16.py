import pytest
import torch

from kernel import gemm_fp16_16_8_8_cuda
from test_framework.test_abc import TestAbc


def gemm_golden(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
) -> None:
    c.copy_(torch.matmul(a, b.T))


class TestGemmCutlass(TestAbc):
    @pytest.mark.parametrize("shape", [(16, 8, 8)])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_gemm_f16(self, shape, dtype):
        m, k, n = shape
        a = self.get_tensor((m, k), dtype, data=torch.arange(m * k, dtype=dtype) % 8)
        b = self.get_tensor((k, n), dtype, data=torch.arange(k * n, dtype=dtype) % 8)
        c = torch.zeros((m, n), device=a.device, dtype=dtype)

        func_args = {}
        self.invoke(
            [a, b],
            [c],
            func_args=func_args,
            kernel_func=gemm_fp16_16_8_8_cuda,
            golden_func=gemm_golden,
        )
