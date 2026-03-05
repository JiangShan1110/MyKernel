#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>
#include <cute/util/print.hpp>

template <typename Spec>
__global__ void gemm_fp16_16_8_8(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k) {
  using namespace cute;

  using X = Underscore;
  using T = typename Spec::T;
  using TiledMMA = typename Spec::TiledMMA;

  constexpr int kTileM = Spec::kTileM;
  constexpr int kTileN = Spec::kTileN;
  constexpr int kTileK = Spec::kTileK;

  int tid = threadIdx.x;

  Tensor mA = make_tensor(
    make_gmem_ptr((T *)Aptr), 
    make_shape(m, k), 
    make_stride(k, Int<1>{})
  ); 

  Tensor mB = make_tensor(
    make_gmem_ptr((T *)Bptr), 
    make_shape(n, k), 
    make_stride(k, Int<1>{})
  );
  
  Tensor mC = make_tensor(
    make_gmem_ptr((T *)Cptr), 
    make_shape(m, n), 
    make_stride(n, Int<1>{})
  );

  auto tiler = make_tile(Int<kTileM>{}, Int<kTileN>{}, Int<kTileK>{});
  auto coord = make_coord(0, 0, 0);
  
  // Define the block global tensors (static)
  Tensor gA = local_tile(
    mA, 
    tiler, 
    coord, 
    Step<_1, X, _1>{}
  ); 

  Tensor gB = local_tile(
    mB, 
    tiler, 
    coord, 
    Step<X, _1, _1>{}
  );
  
  Tensor gC = local_tile(
    mC, 
    tiler, 
    coord, 
    Step<_1, _1, X>{}
  ); 

  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);
 
  Tensor tCgA = thr_mma.partition_A(gA); 
  Tensor tCgB = thr_mma.partition_B(gB);
  Tensor tCgC = thr_mma.partition_C(gC);

  Tensor tCrA = thr_mma.partition_fragment_A(gA); 
  Tensor tCrB = thr_mma.partition_fragment_B(gB);
  Tensor tCrC = thr_mma.partition_fragment_C(gC); 

  auto copy_atom = AutoVectorizingCopy{};

  copy(copy_atom, tCgA, tCrA);
  copy(copy_atom, tCgB, tCrB);

  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  copy(copy_atom, tCrC, tCgC);

  // if (thread0()) {
  //   print_latex(tiled_mma); printf("\n");
  //   print(tCgA); printf("\n");
  //   print(tCgB); printf("\n");
  //   print(tCgC); printf("\n");
  //   print(tCrA); printf("\n");
  //   print(tCrB); printf("\n");
  //   print(tCrC); printf("\n");
  // }
}

namespace spec {

using namespace cute;

template <typename T_, int kTileM_ = 16, int kTileN_ = 8, int kTileK_ = 8> struct KernelSpec {
  using T = T_;

  static constexpr int kTileM = kTileM_;
  static constexpr int kTileN = kTileN_;
  static constexpr int kTileK = kTileK_;

  using MMA_op = SM80_16x8x8_F16F16F16F16_TN;
  using TiledMMA = decltype(make_tiled_mma(MMA_op{}));

  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kShmSize = 0;
};

} // namespace spec

template <typename ComputeType, typename AccType = ComputeType>
void run_minimal_gemm(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &c) {

  at::cuda::CUDAGuard device_guard{a.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  const int M = 16;
  const int N = 8;
  const int K = 8;

  auto torch_compute_type = [] {
    if constexpr (std::is_same_v<ComputeType, cute::half_t>) return torch::kHalf;
    throw std::runtime_error("Unsupported ComputeType!");
  }();

  auto torch_acc_type = [] {
    if constexpr (std::is_same_v<AccType, cute::half_t>) return torch::kHalf;
    throw std::runtime_error("Unsupported AccType!");
  }();

  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "All tensors must be CUDA tensors.");
  TORCH_CHECK(a.dtype() == torch_compute_type, "Tensor a must be of type ", torch_compute_type);
  TORCH_CHECK(b.dtype() == torch_compute_type, "Tensor b must be of type ", torch_compute_type);
  TORCH_CHECK(c.dtype() == torch_acc_type, "Tensor c must be of type ", torch_acc_type);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "All tensors must be 2D.");
  TORCH_CHECK(a.size(0) == M && a.size(1) == K, "Tensor a must be of shape (", M, ", ", K, ").");
  TORCH_CHECK(b.size(0) == N && b.size(1) == K, "Tensor b must be of shape (", N, ", ", K, ").");
  TORCH_CHECK(c.size(0) == M && c.size(1) == N, "Tensor c must be of shape (", M, ", ", N, ").");
  
  using Spec = spec::KernelSpec<ComputeType, M, N, K>;

  // cute::print(typename Spec::TiledMMA{});

  dim3 block = Spec::kThreadNum;
  dim3 grid((N + Spec::kTileN - 1) / Spec::kTileN, (M + Spec::kTileM - 1) / Spec::kTileM);
  int shm_size = Spec::kShmSize;

  printf("Block Size: (%d, %d, %d) | Grid Size: (%d, %d, %d) | Shared Memory Size: %d Bytes\n", block.x, block.y,
         block.z, grid.x, grid.y, grid.z, shm_size);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaDeviceSynchronize();

  // Kernel launch
  cudaEventRecord(start, stream);

  gemm_fp16_16_8_8<Spec><<<grid, block, shm_size, stream>>>(
      reinterpret_cast<AccType *>(c.data_ptr()), reinterpret_cast<ComputeType *>(a.data_ptr()),
      reinterpret_cast<ComputeType *>(b.data_ptr()), M, N, K);

  cudaEventRecord(stop, stream);
  cudaDeviceSynchronize();

  auto error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error) +
                             " (error code: " + std::to_string(error) + ")");
  }
  
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Kernel execution time: %.3f ms\n", milliseconds);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_fp16_16_8_8", &(run_minimal_gemm<cute::half_t>), py::arg("a"), py::arg("b"), py::arg("c"), "Run a single 16x8x8 MMA operation.");
}