import pytest
import torch

from test_framework.test_abc import TestAbc
from test_framework.utils import load_cutlass_extension


def transpose_golden(
    a: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    assert len(a.size()) == 2, "Only support 2D tensor"
    b.copy_(a.swapaxes(-1, -2))


def transpose_cuda(
    a: torch.Tensor,
    b: torch.Tensor | None = None,
    **kwargs,
) -> None:
    ext = load_cutlass_extension(
        "transpose", "kernel/shape/transpose.cu", dump_file=False
    )
    ext.transpose(a, b)


class TestTranspose(TestAbc):
    @pytest.mark.parametrize("shape", [(300, 300)])
    def test_transpose(
        self,
        request,
        shape,
    ):
        a = self.get_tensor(shape, torch.float16)
        b = torch.zeros((shape[1], shape[0]), device=a.device, dtype=a.dtype)

        self.invoke(
            [a],
            [b],
            func_args={},
            kernel_func=transpose_cuda,
            golden_func=transpose_golden,
        )
