
from pathlib import Path
from typing import Optional, Tuple
from tabulate import tabulate
import torch
from torch.utils.cpp_extension import load

from utils import LOG, Col

class TestAbc:
    _golden_func = None
    _cuda_name = ""
    _cuda_sources = []
    _cuda_extra_cuda_cflags = []
    _cuda_extra_cflags = []

    @staticmethod
    def get_tensor(
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        scale: float = 8.0,
        bias: float = -8.0,
    ) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return scale * torch.rand(size=shape, dtype=dtype, device=device) + bias
    
    def _compare_tensors(self, a: torch.Tensor, b: torch.Tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        LOG.debug(f"Comparing {Col.M} <{a.shape}, {a.dtype}> {Col.RESET} v.s. {Col.M} <{b.shape}, {b.dtype}> {Col.RESET} with rtol={rtol}, atol={atol}")

        if a.shape != b.shape:
            raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

        if a.is_floating_point() or b.is_floating_point():
            mask = ~torch.isclose(a, b, rtol=rtol, atol=atol)
        else:
            mask = a != b

        if not mask.any():
            LOG.debug(f"{Col.G} Tensor comparison succeeded. {Col.RESET}")
            return 
        
        idx = torch.nonzero(mask, as_tuple=False)   # [N, ndim]

        vals_a = a[mask].flatten()
        vals_b = b[mask].flatten()

        table = []
        for i in range(idx.size(0)):
            coord = tuple(idx[i].tolist())
            a_val = vals_a[i].item()
            b_val = vals_b[i].item()
            cur_atol = abs(a_val - b_val)
            cur_rtol = cur_atol / (abs(b_val) + 1e-12)
            table.append([coord, a_val, b_val, cur_atol, cur_rtol])

        LOG.debug(
            f"Get {idx.size(0)} different elements:\n{tabulate(table, headers=['index', 'model', 'golden', 'atol', 'rtol'], tablefmt='simple')}",
        )

        raise AssertionError(f"{Col.R}Tensor comparison failed.{Col.RESET}")

    def _kernel_func(self, *tensor: torch.Tensor, **kwargs) -> None:
        func_name = self._cuda_name
        func_file = [Path("./src/kernel") / path for path in self._cuda_sources]
        lib = load(
            name=func_name,
            sources=func_file,
            extra_cuda_cflags=self._cuda_extra_cuda_cflags,
            extra_cflags=self._cuda_extra_cflags,
        )
        func = getattr(lib, func_name)
        func(*tensor, **kwargs)

    def invoke(
        self,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
        attrs: dict,
        **kwargs,
    ):  
        golden_output = torch.empty_like(output)
        self._kernel_func(*inputs, output, **attrs)
        self._golden_func(*inputs, golden_output, **attrs)
        self._compare_tensors(output, golden_output, rtol=kwargs.get("rtol", 1e-5), atol=kwargs.get("atol", 1e-8))

        return output, golden_output

    

