#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>
#include <cute/util/print.hpp>

#define TIME_CUDA_KERNEL(stream, ...)                                   \
    do {                                                                \
        cudaEvent_t __start, __stop;                                    \
        cudaEventCreate(&__start);                                      \
        cudaEventCreate(&__stop);                                       \
        cudaDeviceSynchronize();                                        \
                                                                        \
        cudaEventRecord(__start, stream);                               \
        {                                                               \
            __VA_ARGS__;                                                \
        }                                                               \
        cudaEventRecord(__stop, stream);                                \
        cudaDeviceSynchronize();                                        \
                                                                        \
        auto __error = cudaGetLastError();                              \
        if (__error != cudaSuccess) {                                   \
            cudaEventDestroy(__start);                                  \
            cudaEventDestroy(__stop);                                   \
            throw std::runtime_error(                                   \
                std::string("CUDA error: ") + cudaGetErrorString(__error) + \
                " (error code: " + std::to_string(__error) + ")");  \
        }                                                               \
                                                                        \
        float __milliseconds = 0;                                       \
        cudaEventElapsedTime(&__milliseconds, __start, __stop);         \
        printf("Kernel execution time: %.3f ms\n", __milliseconds);     \
                                                                        \
        cudaEventDestroy(__start);                                      \
        cudaEventDestroy(__stop);                                       \
    } while(0)

template <typename Config>
__global__ void gemm_fp16_16_8_8(void *Cptr, const void *Aptr, const void *Bptr, int m, int n, int k) {
  using namespace cute;

  using X = Underscore;
  using T = typename Config::T;
  using TiledMMA = typename Config::TiledMMA;

  constexpr int kBlockM = Config::kBlockM;
  constexpr int kBlockN = Config::kBlockN;
  constexpr int kBlockK = Config::kBlockK;
  
  // In software, 128 thread
  // In hardware, 4 warps, each warp has 32 threads
  int tid = threadIdx.x;
 
  // tensor layout
  Tensor mA = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k), make_stride(k, Int<1>{})); 
  Tensor mB = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr((T *)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
  auto coord = make_coord(0, 0, 0);
  
  // the data that the block will process
  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); 
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); 

  // thread
  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);
  
  // the data of glm that the thread will process
  Tensor tCgA = thr_mma.partition_A(gA); 
  Tensor tCgB = thr_mma.partition_B(gB);
  Tensor tCgC = thr_mma.partition_C(gC);

  // the register that the thread will use to store the data
  Tensor tCrA = thr_mma.partition_fragment_A(gA); 
  Tensor tCrB = thr_mma.partition_fragment_B(gB);
  Tensor tCrC = thr_mma.partition_fragment_C(gC); 

  auto copy_atom = AutoVectorizingCopy{};

  // load data from glm to register
  copy(copy_atom, tCgA, tCrA);
  copy(copy_atom, tCgB, tCrB);

  gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

  copy(copy_atom, tCrC, tCgC);
}

namespace Config {

using namespace cute;

template <typename T_> struct KernelSpec {
  using T = T_;

  using MMAInstr = SM80_16x8x8_F16F16F16F16_TN;
  using MMA_traits = MMA_Traits<MMAInstr>;
  using MMA_atom = MMA_Atom<MMA_traits>;
  using MMA_shape = MMA_traits::Shape_MNK;

  constexpr static int kTiledM = size<0>(MMA_shape{});
  constexpr static int kTiledN = size<1>(MMA_shape{});
  constexpr static int kTiledK = size<2>(MMA_shape{});

  constexpr static int kLoopM = 2;
  constexpr static int kLoopN = 4;
  constexpr static int kLoopK = 2;
  
  // 32 32 16
  constexpr static int kWarpTiledM = kTiledM * kLoopM;
  constexpr static int kWarpTiledN = kTiledN * kLoopN;
  constexpr static int kWarpTiledK = kTiledK * kLoopK;

  constexpr static int kWarpM = 2;
  constexpr static int kWarpN = 2;
  constexpr static int kWarpK = 1;
  
  // 64 64 16
  constexpr static int kBlockTiledM = kWarpM * kWarpTiledM;
  constexpr static int kBlockTiledN = kWarpN * kWarpTiledN;
  constexpr static int kBlockTiledK = kWarpK * kWarpTiledK;

  // for m in (2)
  //  for n in (2)
  //    for k int (2)
  //      Tiled MMA
  constexpr static int kBlockM = kBlockTiledM * 2;
  constexpr static int kBlockN = kBlockTiledN * 2;
  constexpr static int kBlockK = kBlockTiledK * 2;

  
  // warp nums
  using MMAThrLayout = decltype(make_layout(make_shape(Int<kWarpM>{}, Int<kWarpN>{}, Int<kWarpK>{}),
                                            make_stride(Int<kWarpN * kWarpK>{}, Int<kWarpK>{}, Int<1>{})));
  // tiled per block
  using Permutations = Tile<Int<kBlockTiledM>, Int<kBlockTiledN>, Int<kBlockTiledK>>;
  using TiledMMA = decltype(make_tiled_mma(MMAInstr{}, MMAThrLayout{}, Permutations{}));


  using Copy_G2S_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;

  static constexpr int kWarpThrNum = 32;
  // thread nums per block
  static constexpr int kThreadNum = size(TiledMMA{});
  static constexpr int kShmSize = 0;
};

} // namespace Config

template <typename ComputeType, typename AccType = ComputeType>
void run_gemm(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &c) {

  at::cuda::CUDAGuard device_guard{a.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

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
  TORCH_CHECK(c.dtype() == torch_compute_type, "Tensor c must be of type ", torch_compute_type);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "All tensors must be 2D.");
  
  using Config = typename Config::template KernelSpec<ComputeType>;

  static constexpr int M = Config::kBlockM;
  static constexpr int N = Config::kBlockN;
  static constexpr int K = Config::kBlockK;

  TORCH_CHECK(a.size(0) == M && a.size(1) == K, "Tensor a must be of shape (", M, ", ", K, ").");
  TORCH_CHECK(b.size(0) == N && b.size(1) == K, "Tensor b must be of shape (", N, ", ", K, ").");
  TORCH_CHECK(c.size(0) == M && c.size(1) == N, "Tensor c must be of shape (", M, ", ", N, ").");

  // cute::print(typename Config::TiledMMA{});

  dim3 block(Config::kThreadNum, 1, 1);
  dim3 grid(1, 1, 1);
  int shm_size = Config::kShmSize;

  TIME_CUDA_KERNEL(stream,
    gemm_fp16_16_8_8<Config><<<grid, block, shm_size, stream>>>(
      reinterpret_cast<AccType *>(c.data_ptr()), reinterpret_cast<ComputeType *>(a.data_ptr()),
      reinterpret_cast<ComputeType *>(b.data_ptr()), M, N, K)
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_fp16_16_8_8", &(run_gemm<cute::half_t>), py::arg("a"), py::arg("b"), py::arg("c"), "Run a single 16x8x8 MMA operation.");
}