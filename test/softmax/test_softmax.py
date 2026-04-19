import pytest
import torch

from test_framework.test_abc import TestAbc
from test_framework.utils import load_cutlass_extension


def softmax_golden(
    a: torch.Tensor,
    b: torch.Tensor | None = None,
) -> torch.Tensor:
    assert len(a.size()) == 1, "Only support 1D tensor"
    from torch import nn

    b.copy_(nn.functional.softmax(a, dim=0))


def softmax_cuda(
    a: torch.Tensor,
    b: torch.Tensor | None = None,
    **kwargs,
) -> None:
    ext = load_cutlass_extension(
        "softmax", "kernel/softmax/softmax.cu", dump_file=False
    )
    ext.softmax(a, b)


class TestSoftmax(TestAbc):
    @pytest.mark.parametrize("shape", [(300,)])
    def test_softmax(
        self,
        request,
        shape,
    ):
        a = self.get_tensor(shape, torch.float32)
        b = torch.zeros_like(a)

        self.invoke(
            [a],
            [b],
            func_args={},
            kernel_func=softmax_cuda,
            golden_func=softmax_golden,
        )
