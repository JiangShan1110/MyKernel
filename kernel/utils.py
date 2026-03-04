from functools import lru_cache
from pathlib import Path

from torch.utils.cpp_extension import load


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
        ],
        verbose=False,
    )
