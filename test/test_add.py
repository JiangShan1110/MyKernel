import pytest
import torch
from test_framework import TestAbc

def add(*tensors: torch.Tensor, **attrs) -> None:
    if len(tensors) != 3:
        raise ValueError(f"Expect 3 tensors (a, b, c) for torch.add operation, but got {tensors}.")
    a, b, c = tensors
    # torch.add: out_of_place
    c.copy_(a + b)

class TestAdd(TestAbc):
    _golden_func = staticmethod(add)
    _cuda_name = "elementwise_add_f32"
    _cuda_sources = ["elementwise/elementwise.cu"]
    
    @pytest.mark.parametrize("shape", [(16, 32), (64, 128), (128, 256)])
    def test_add(self, shape):
        a = self.get_tensor(shape)
        b = self.get_tensor(shape)
        output = torch.empty_like(a)
        self.invoke((a, b), output, attrs={})