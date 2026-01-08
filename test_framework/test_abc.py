import os
from typing import Optional, Tuple

import torch
from tabulate import tabulate

from .utils import LOG


class TestAbc:
    _atol = 1e-2
    _rtol = 1e-2

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
            return torch.arange(
                0, torch.prod(torch.tensor(shape)), dtype=dtype, device=device
            ).reshape(shape)
        else:
            return scale * torch.rand(size=shape, dtype=dtype, device=device) + bias

    def _compare_tensors(
        self,
        model_res: torch.Tensor,
        golden_res: torch.Tensor,
        rtol: float,
        atol: float,
    ) -> bool:
        LOG.info(
            "Comparing Model (%s, %s) v.s. Golden (%s, %s)",
            model_res.shape,
            model_res.dtype,
            golden_res.shape,
            golden_res.dtype,
        )

        if model_res.shape != golden_res.shape:
            raise ValueError(f"shape mismatch: {model_res.shape} vs {golden_res.shape}")
        if model_res.dtype != golden_res.dtype:
            raise ValueError(f"dtype mismatch: {model_res.dtype} vs {golden_res.dtype}")

        if model_res.is_floating_point() or golden_res.is_floating_point():
            mask = ~torch.isclose(model_res, golden_res, rtol=rtol, atol=atol)
        else:
            mask = model_res != golden_res

        err_count = mask.sum().item()
        total = model_res.numel()
        ratio = err_count / total * 100 if total else 0.0

        LOG.debug("Error ratio: %.4f%% (rtol=%g, atol=%g)", ratio, rtol, atol)
        if err_count == 0:
            return True

        idx = torch.nonzero(mask, as_tuple=False)  # [N, ndim]
        vals_a = model_res[mask]  # 1D tensor
        vals_b = golden_res[mask]

        atol_vec = (vals_a - vals_b).abs()
        rtol_vec = atol_vec / (vals_b.abs() + 1e-12)

        _, top_i = torch.topk(atol_vec, k=min(50, err_count))
        top_idx = idx[top_i].tolist()
        top_a = vals_a[top_i].cpu().numpy()
        top_b = vals_b[top_i].cpu().numpy()
        top_atol = atol_vec[top_i].cpu().numpy()
        top_rtol = rtol_vec[top_i].cpu().numpy()

        table = [
            (idx, a, b, atol, rtol)
            for idx, a, b, atol, rtol in zip(top_idx, top_a, top_b, top_atol, top_rtol)
        ]
        print_table = tabulate(
            table,
            headers=["index", "model", "golden", "atol", "rtol"],
            colalign=("left",) * 5,
        )
        LOG.error("Tensor comparison failed!")
        LOG.debug("Top-50 error elements:\n%s", print_table, extra={"indent": ""})
        raise AssertionError("Tensor comparison failed.")

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
                rtol=kwargs.get("rtol", self._rtol),
                atol=kwargs.get("atol", self._atol),
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
