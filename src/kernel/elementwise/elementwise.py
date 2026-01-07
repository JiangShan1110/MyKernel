import triton
import triton.language as tl
import torch

@triton.jit
def elementwise_add_kernel(
    x_ptr, y_ptr, z_ptr,
    N: tl.constexpr, BLOCK_SIZE: tl.constexpr=1024
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offset = start + tl.arange(0, BLOCK_SIZE)
    mask = offset < N

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)
    z = x + y
    tl.store(z_ptr + offset, z, mask=mask)
    

def elementwise_add(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, BLOCK_SIZE: int=1024) -> torch.Tensor:
    assert x.is_cuda and y.is_cuda and x.shape == y.shape
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    elementwise_add_kernel[grid](x, y, z, N, BLOCK_SIZE=BLOCK_SIZE)
    return z