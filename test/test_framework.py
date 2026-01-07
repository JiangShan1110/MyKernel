import os
from typing import Optional, Tuple

import torch
from tabulate import tabulate
from utils import LOG


class TestAbc:
    @staticmethod
    def get_tensor(
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: Optional[torch.device] = None,
        scale: float = 8.0,
        bias: float = -8.0,
        is_random: bool = True,
        data: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if data is not None:
            return data.to(device=device, dtype=dtype).reshape(shape)
        
        if not is_random:
            return torch.arange(0, torch.prod(torch.tensor(shape)), dtype=dtype, device=device).reshape(shape)
        else:
            return scale * torch.rand(size=shape, dtype=dtype, device=device) + bias

    def _compare_tensors(
        self,
        model_res: torch.Tensor,
        golden_res: torch.Tensor,
        rtol: float = 1e-5,
        atol: float = 1e-8,
    ) -> bool:
        LOG.info(
            f"Comparing Model ({model_res.shape}, {model_res.dtype}) v.s. Golden ({golden_res.shape}, {golden_res.dtype})"
        )

        if model_res.shape != golden_res.shape:
            raise ValueError(f"shape mismatch: {model_res.shape} vs {golden_res.shape}")

        if model_res.is_floating_point() or golden_res.is_floating_point():
            mask = ~torch.isclose(model_res, golden_res, rtol=rtol, atol=atol)
        else:
            mask = model_res != golden_res

        idx = torch.nonzero(mask, as_tuple=False)  # [N, ndim]
        ratio = mask.sum().item() / idx.size(0) * 100 if idx.size(0) > 0 else 0.0
        LOG.debug(f"Get error ratio: {ratio:.2f}% with rtol: {rtol}, atol: {atol}.")

        if not mask.any():
            LOG.debug(f"Tensor comparison succeeded.")
            return

        vals_a = model_res[mask].flatten()
        vals_b = golden_res[mask].flatten()

        table = []
        for i in range(idx.size(0)):
            coord = idx[i].tolist()
            a_val = vals_a[i].item()
            b_val = vals_b[i].item()
            cur_atol = abs(a_val - b_val)
            cur_rtol = cur_atol / (abs(b_val) + 1e-12)
            table.append([coord, a_val, b_val, cur_atol, cur_rtol])

        table_sorted = sorted(table, key=lambda r: r[3], reverse=True)[:50]
        print_table = tabulate(
            table_sorted,
            headers=["index", "model", "golden", "atol", "rtol"],
            tablefmt="grid",
            colalign=("left",) * 5,
        )
        LOG.error(f"Tensor comparison failed!")
        LOG.error(
            f"Top-50 error elements:\n{print_table}",
        )

        raise AssertionError(f"Tensor comparison failed.")

    def invoke(
        self,
        inputs: Tuple[torch.Tensor, ...],
        outputs: Tuple[torch.Tensor, ...],
        attrs: dict,
        kernel_func: callable = None,
        golden_func: callable = None,
        **kwargs,
    ):
        LOG.info(f"Current PID: {os.getpid()}")
        LOG.info(f"Get inputs {[ (t.shape, t.dtype) for t in inputs ]}")
        LOG.info(f"Get output: {[ (t.shape, t.dtype) for t in outputs ]}")
        LOG.info(f"Get attrs: {attrs}")

        golden_outputs = [torch.empty_like(out) for out in outputs]

        LOG.info("Running golden function...")
        golden_func(*inputs, *golden_outputs, **attrs)

        LOG.info("Running kernel function...")
        kernel_func(*inputs, *outputs, **attrs)

        LOG.info("Comparing outputs...")
        for output, golden_output in zip(outputs, golden_outputs):
            self._compare_tensors(
                output.detach().cpu(),
                golden_output.detach().cpu(),
                rtol=kwargs.get("rtol", 1e-5),
                atol=kwargs.get("atol", 1e-8),
            )

        return output, golden_output

    def perf(
        self,
        inputs: Tuple[torch.Tensor, ...],
        outputs: Tuple[torch.Tensor, ...],
        attrs: dict,
        kernel_func: callable = None,
        golden_func: callable = None,
        repeat: int = 100,
    ) -> float:
        LOG.info(f"Current PID: {os.getpid()}")
        LOG.info(f"Get inputs {[ (t.shape, t.dtype) for t in inputs ]}")
        LOG.info(f"Get output: {[ (t.shape, t.dtype) for t in outputs ]}")
        LOG.info(f"Get attrs: {attrs}")

        # Warm up
        self.kernel_func(*inputs, *outputs, **attrs)
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            self.kernel_func(*inputs, *outputs, **attrs)
        end.record()
        torch.cuda.synchronize()
        elapsed_time_ms = start.elapsed_time(end) / repeat
        LOG.info(f"Average execution time: {elapsed_time_ms:.6f} ms")
        return elapsed_time_ms