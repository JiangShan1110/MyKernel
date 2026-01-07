import pytest
import torch
from test_framework import TestAbc

def add(*tensors: torch.Tensor, **attrs) -> None:
    if len(tensors) != 3:
        raise ValueError(
            f"Expect 3 tensors (a, b, c) for torch.add operation, but got {len(tensors)}: {tensors}."
        )
    a, b, c = tensors
    # torch.add: out_of_place
    c.copy_(a + b)

class TestAdd(TestAbc):
    @pytest.mark.parametrize("shape", [(16, 32), (64, 128), (128, 256)])
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_add(
        self,
        shape,
        dtype,
    ):
        a = self.get_tensor(shape, dtype)
        b = self.get_tensor(shape, dtype)
        output = torch.empty_like(a)

        from torch.utils.cpp_extension import load
        func_name = "elementwise_add_f32"
        func_file = ["src/kernel/elementwise/elementwise.cu"]
        lib = load(
            name=func_name,
            sources=func_file,
            extra_cuda_cflags=[
                "-O3",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
            extra_cflags=["-std=c++17"],
        )
        func = getattr(lib, func_name)

        self.invoke([a, b], [output], attrs={}, kernel_func=func, golden_func=add)

    @pytest.mark.parametrize("shape", [(1024, 2048), (2048, 4096)])
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_add_triton(
        self,
        shape,
        dtype,
    ):
        from src.kernel.elementwise.elementwise import elementwise_add
        a = self.get_tensor(shape, dtype)
        b = self.get_tensor(shape, dtype)
        output = torch.empty_like(a)
        attrs = {"BLOCK_SIZE": 1024}
        self.invoke([a, b], [output], attrs=attrs, kernel_func=elementwise_add, golden_func=add)
