import pytest
import torch

from test_framework.test_abc import TestAbc


def gemm_golden(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.0,
) -> None:
    c[::] = alpha * (a @ b) + beta * c


class TestGemmCutlass(TestAbc):
    @pytest.mark.parametrize("shape", [(128, 128, 128), (256, 128, 256)])
    @pytest.mark.parametrize("dtype", [torch.float16])
    def test_gemm_f16(self, shape, dtype):
        from kernel import gemm_f16_cuda

        m, k, n = shape
        a = self.get_tensor((m, k), dtype)
        b = self.get_tensor((k, n), dtype)
        c = torch.zeros((m, n), device=a.device, dtype=dtype)

        func_args = {"alpha": 1.0, "beta": 0.0}
        self.invoke(
            [a, b],
            [c],
            func_args=func_args,
            kernel_func=gemm_f16_cuda,
            golden_func=gemm_golden,
        )
