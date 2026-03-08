import datetime
import logging
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load

LOG = logging.getLogger(__name__)
DIR = Path("/root/MyKernel")
ARCH = ".".join(map(str, torch.cuda.get_device_capability()))
os.environ["TORCH_CUDA_ARCH_LIST"] = ARCH


def make_json_friendly(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, (list, tuple)):
        return [make_json_friendly(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): make_json_friendly(v) for k, v in obj.items()}
    return str(obj)


def load_cutlass_extension(
    name: str,
    source_file: str,
    dump_file: bool = False,
) -> Any:
    cutlass_root = DIR / "third_party" / "cutlass"
    include_dir = cutlass_root / "include"
    util_include_dir = cutlass_root / "tools" / "util" / "include"
    file_path = DIR / source_file

    cflags = [
        "-O3",
        "--use_fast_math",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ftemplate-backtrace-limit=0",  # To debug template code
        "--resource-usage",  # printing out number of registers
        # "--ptxas-options=--verbose,--register-usage-level=5,--warn-on-local-memory-usage",  # printing out number of registers
        "--generate-line-info",  # show PTX and SASS in ncu
        "-DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED",
        "-DCUTLASS_ENABLE_GDC_FOR_SM90",  # For PDL
        "-DCUTLASS_ENABLE_GDC_FOR_SM100",  # For PDL
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",  # Can toggle for debugging
        "-DNDEBUG",  # Important, otherwise performance is severely impacted
        "-Xfatbin",  # compress all binary sections
        "-compress-all",
        # for debug purpose
        # "-G", # device debug
        # "-g", # host debug
        # "-Xcompiler",
        # "-rdynamic",
    ]

    if dump_file:
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        build_dir = DIR / "build" / name / time_str
        os.makedirs(build_dir, exist_ok=True)
        cflags.extend(["-keep", f"--keep-dir {build_dir}"])

    return load(
        name=name,
        sources=[str(file_path)],
        extra_include_paths=[
            str(include_dir),
            str(util_include_dir),
        ],
        extra_cflags=["-std=c++17"],
        verbose=True,  # show compile logs
        extra_cuda_cflags=cflags,
    )
