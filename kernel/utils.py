import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = ".".join(
    map(str, torch.cuda.get_device_capability())
)


@lru_cache(maxsize=1)
def load_cutlass_extension(name, source_file):
    cutlass_root = Path("third_party") / "cutlass"
    include_dir = cutlass_root / "include"
    util_include_dir = cutlass_root / "tools" / "util" / "include"

    return load(
        name=name,
        sources=[str(source_file)],
        extra_include_paths=[str(include_dir), str(util_include_dir)],
        extra_cflags=["-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
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
        ],
        verbose=False,
    )
