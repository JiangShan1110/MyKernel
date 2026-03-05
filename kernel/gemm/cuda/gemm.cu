#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <torch/extension.h>

using RowMajor = cutlass::layout::RowMajor;
using ElementInput = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using ElementAccumulator = float;

namespace {

void check_cuda_tensor(const torch::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dtype() == torch::kFloat16, name, " must be float16");
}

}  // namespace

void gemm_f16(torch::Tensor a, torch::Tensor b, torch::Tensor c, float alpha,
              float beta) {
  check_cuda_tensor(a, "a");
  check_cuda_tensor(b, "b");
  check_cuda_tensor(c, "c");

  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2,
              "a, b, c must be 2D tensors");

  const int64_t m = a.size(0);
  const int64_t k = a.size(1);
  const int64_t n = b.size(1);

  TORCH_CHECK(b.size(0) == k, "shape mismatch: b.size(0) must equal a.size(1)");
  TORCH_CHECK(c.size(0) == m && c.size(1) == n,
              "shape mismatch: c must be [a.size(0), b.size(1)]");
  TORCH_CHECK(alpha == 1.0f);
  TORCH_CHECK(beta == 0.0f);

  using Gemm = cutlass::gemm::device::Gemm<ElementInput, RowMajor, ElementInput,
                                           RowMajor, ElementOutput, RowMajor,
                                           ElementAccumulator>;

  Gemm gemm_op;

  const int lda = static_cast<int>(k);
  const int ldb = static_cast<int>(n);
  const int ldc = static_cast<int>(n);

  typename Gemm::Arguments args(
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)},
      {reinterpret_cast<ElementInput *>(a.data_ptr()), lda},
      {reinterpret_cast<ElementInput *>(b.data_ptr()), ldb},
      {reinterpret_cast<ElementOutput *>(c.data_ptr()), ldc},
      {reinterpret_cast<ElementOutput *>(c.data_ptr()), ldc},
      {alpha, beta});

  cutlass::Status status = gemm_op(args);
  TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM launch failed");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
    "gemm_f16", 
    &gemm_f16, 
    pybind11::arg("a"), 
    pybind11::arg("b"),
    pybind11::arg("c"), 
    pybind11::arg("alpha") = 1.0f,
    pybind11::arg("beta") = 0.0f,
    "Run Cutlass Tmpl Gemm"
  );
}
