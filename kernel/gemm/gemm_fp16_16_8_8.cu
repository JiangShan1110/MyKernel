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
  using TiledCopyAG2S = typename Config::TiledCopyAG2S;
  using TiledCopyBG2S = typename Config::TiledCopyBG2S;
  using TiledCopyAS2R = typename Config::TiledCopyAS2R;
  using TiledCopyBS2R = typename Config::TiledCopyBS2R;
  using SmemALayout = typename Config::SmemALayout;
  using SmemBLayout = typename Config::SmemBLayout;
  using TiledCopyCR2G = typename Config::TiledCopyCR2G;

  constexpr int kBlockM = Config::kBlockTiledM;
  constexpr int kBlockN = Config::kBlockTiledN;
  constexpr int kBlockK = Config::kBlockTiledK;
  
  // In software, 128 thread
  // In hardware, 4 warps, each warp has 32 threads
  int tid = threadIdx.x;
 
  // tensor layout
  Tensor mA = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k), make_stride(k, Int<1>{})); 
  Tensor mB = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
  Tensor mC = make_tensor(make_gmem_ptr((T *)Cptr), make_shape(m, n), make_stride(n, Int<1>{}));

  auto tiler = make_tile(Int<kBlockM>{}, Int<kBlockN>{}, Int<kBlockK>{});
  int m_idx = blockIdx.x;
  int n_idx = blockIdx.y;
  auto coord = make_coord(m_idx, n_idx, _);

  int loop_k = k / kBlockK;
  
  // the data that the block will process
  Tensor gA = local_tile(mA, tiler, coord, Step<_1, X, _1>{}); // (128, 32), loop_k
  Tensor gB = local_tile(mB, tiler, coord, Step<X, _1, _1>{}); // (128, 32), loop_k
  Tensor gC = local_tile(mC, tiler, coord, Step<_1, _1, X>{}); // (128, 128)

  extern __shared__ __align__(1024) uint8_t smem[];
  uint8_t *ptr_sA = smem;
  uint8_t *ptr_sB = smem + Config::kSmemSizeA;

  Tensor sA = make_tensor(make_smem_ptr<T>(ptr_sA), SmemALayout{}); // (128, 32)
  Tensor sB = make_tensor(make_smem_ptr<T>(ptr_sB), SmemBLayout{}); // (128, 32)

  // thread
  TiledMMA tiled_mma;
  ThrMMA thr_mma = tiled_mma.get_slice(tid);
  

  // the register that the thread will use to store the data
  Tensor tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));  // (64, 16), 2, 2
  Tensor tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));  // (64, 16), 2, 2
  Tensor tCrC = thr_mma.partition_fragment_C(gC);  // (64, 64), 2, 2


  // copy data from glm to smem
  TiledCopyAG2S tiled_copy_a_g2s;
  ThrCopy thr_copy_a_g2s = tiled_copy_a_g2s.get_slice(tid);
  // src glm
  Tensor tAgA_g2s = thr_copy_a_g2s.partition_S(gA); // (32, 32), 4, 1, loop_k
  // dst smem
  Tensor tAsA_g2s = thr_copy_a_g2s.partition_D(sA); // (32, 32), 4, 1

  TiledCopyBG2S tiled_copy_b_g2s;
  ThrCopy thr_copy_b_g2s = tiled_copy_b_g2s.get_slice(tid);
  Tensor tBgB_g2s = thr_copy_b_g2s.partition_S(gB); // (32, 32), 4, 1, loop_k
  Tensor tBsB_g2s = thr_copy_b_g2s.partition_D(sB); // (32, 32), 4, 1

  TiledCopyAS2R tiled_copy_a_s2r;
  ThrCopy thr_copy_a_s2r = tiled_copy_a_s2r.get_slice(tid);
  Tensor tAsA_s2r = thr_copy_a_s2r.partition_S(sA); // (64, 16), 2, 2
  Tensor tCrA_s2r = thr_copy_a_s2r.retile_D(tCrA); // (64, 16), 2, 2

  TiledCopyBS2R tiled_copy_b_s2r;
  ThrCopy thr_copy_b_s2r = tiled_copy_b_s2r.get_slice(tid);
  Tensor tBsB_s2r = thr_copy_b_s2r.partition_S(sB);
  Tensor tCrB_s2r = thr_copy_b_s2r.retile_D(tCrB);

  TiledCopyCR2G tiled_copy_c_r2g;
  ThrCopy thr_copy_c_r2g = tiled_copy_c_r2g.get_slice(tid);
  Tensor tCrC_r2g = thr_copy_c_r2g.retile_S(tCrC);
  Tensor tCgC_r2g = thr_copy_c_r2g.partition_D(gC);

  clear(tCrC);

  for (int itile = 0; itile < loop_k; ++itile) {
    // copy a tile of A and B from global memory to shared memory
    copy(tiled_copy_a_g2s, tAgA_g2s(_,_,_,itile), tAsA_g2s);
    copy(tiled_copy_b_g2s, tBgB_g2s(_,_,_,itile), tBsB_g2s);

    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    copy(tiled_copy_a_s2r, tAsA_s2r, tCrA_s2r);
    copy(tiled_copy_b_s2r, tBsB_s2r, tCrB_s2r);

    gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

    copy(tiled_copy_c_r2g, tCrC_r2g, tCgC_r2g);
  }
}

namespace Config {

using namespace cute;

template <typename T_> struct KernelSpec {
  using T = T_;

  using MMAInstr = SM80_16x8x8_F16F16F16F16_TN;
  using MMATraits = MMA_Traits<MMAInstr>;
  using MMAAtom = MMA_Atom<MMATraits>;
  using MMAShape = MMATraits::Shape_MNK;

  constexpr static int kTiledM = size<0>(MMAShape{});
  constexpr static int kTiledN = size<1>(MMAShape{});
  constexpr static int kTiledK = size<2>(MMAShape{});

  constexpr static int kLoopM = 2;
  constexpr static int kLoopN = 4;
  constexpr static int kLoopK = 2;
  
  // warp tiled: 32 32 16
  constexpr static int kWarpTiledM = kTiledM * kLoopM;
  constexpr static int kWarpTiledN = kTiledN * kLoopN;
  constexpr static int kWarpTiledK = kTiledK * kLoopK;

  constexpr static int kWarpM = 2;
  constexpr static int kWarpN = 2;
  constexpr static int kWarpK = 1;
  
  // Tiled mma: 64 64 16
  constexpr static int kTiledMmaM = kWarpM * kWarpTiledM; // 64
  constexpr static int kTiledMmaN = kWarpN * kWarpTiledN; // 64
  constexpr static int kTiledMmaK = kWarpK * kWarpTiledK; // 16
  
  // block tiled: 128 128 32
  constexpr static int kBlockTiledM = kTiledMmaM * 2;
  constexpr static int kBlockTiledN = kTiledMmaN * 2;
  constexpr static int kBlockTiledK = kTiledMmaK * 2;

  
  // warp nums
  using MMAThrLayout = decltype(make_layout(make_shape(Int<kWarpM>{}, Int<kWarpN>{}, Int<kWarpK>{}),
                                            make_stride(Int<kWarpN * kWarpK>{}, Int<kWarpK>{}, Int<1>{})));
  // size of tiled mma 
  using Permutations = Tile<Int<kTiledMmaM>, Int<kTiledMmaN>, Int<kTiledMmaK>>;
  using TiledMMA = decltype(make_tiled_mma(MMAInstr{}, MMAThrLayout{}, Permutations{}));

  static constexpr int kWarpThrNum = 32;
  // thread nums per block
  static constexpr int kThreadNum = size(TiledMMA{});


  // CACHEALWAYS: data will be cached in L1 cache
  // CACHEGLOBAL: bypassing L1 cache
  // fp16 and instr is e8 -> uint128_t
  using CopyG2SInstr = SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>;
  using CopyAG2SAtom = Copy_Atom<CopyG2SInstr, T>;
  using CopyBG2SAtom = Copy_Atom<CopyG2SInstr, T>;

  static constexpr int kElement = 8;
  static constexpr int kBlockCopyKThr = kBlockTiledK / kElement; // 32/8=4
  
  using TiledCopyAG2S = decltype(make_tiled_copy(CopyAG2SAtom{}, 
                                                make_layout(make_shape(Int<kThreadNum / kBlockCopyKThr>{}, Int<kBlockCopyKThr>{}),
                                                              make_stride(Int<kBlockCopyKThr>{}, Int<1>{})),
                                                make_layout(make_shape(Int<1>{}, Int<kElement>{}),
                                                          make_stride(Int<kElement>{}, Int<1>{}))));
  using TiledCopyBG2S = decltype(make_tiled_copy(CopyBG2SAtom{}, 
                                                make_layout(make_shape(Int<kThreadNum / kBlockCopyKThr>{}, Int<kBlockCopyKThr>{}),
                                                              make_stride(Int<kBlockCopyKThr>{}, Int<1>{})),
                                                make_layout(make_shape(Int<1>{}, Int<kElement>{}),
                                                          make_stride(Int<kElement>{}, Int<1>{}))));
                                                          
  using SmemALayout = decltype(make_layout(make_shape(Int<kBlockTiledM>{}, Int<kBlockTiledK>{}),
                                  make_stride(Int<kBlockTiledK>{}, Int<1>{})));
  using SmemBLayout = decltype(make_layout(make_shape(Int<kBlockTiledN>{}, Int<kBlockTiledK>{}),
                                  make_stride(Int<kBlockTiledK>{}, Int<1>{})));
  
  static constexpr int kSmemSizeA = cosize(SmemALayout{}) * sizeof(T);
  static constexpr int kSmemSizeB = cosize(SmemBLayout{}) * sizeof(T);
  static constexpr int kShmSize = kSmemSizeA + kSmemSizeB;

  using CopyAS2RAtom = Copy_Atom<AutoVectorizingCopy, T>;
  using TiledCopyAS2R = decltype(make_tiled_copy_A(CopyAS2RAtom{}, TiledMMA{}));
  using CopyBS2RAtom = Copy_Atom<AutoVectorizingCopy, T>;
  using TiledCopyBS2R = decltype(make_tiled_copy_B(CopyBS2RAtom{}, TiledMMA{}));
  using CopyCR2GAtom = Copy_Atom<AutoVectorizingCopy, T>;
  using TiledCopyCR2G = decltype(make_tiled_copy_C(CopyCR2GAtom{}, TiledMMA{}));
};

} // namespace Config

void run_gemm(const torch::Tensor &a, const torch::Tensor &b, torch::Tensor &c) {

  at::cuda::CUDAGuard device_guard{a.get_device()};
  auto stream = at::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "All tensors must be CUDA tensors.");
  TORCH_CHECK(a.dtype() == torch::kHalf, "Tensor a must be of type ", torch::kHalf);
  TORCH_CHECK(b.dtype() == torch::kHalf, "Tensor b must be of type ", torch::kHalf);
  TORCH_CHECK(c.dtype() == torch::kHalf, "Tensor c must be of type ", torch::kHalf);
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2, "All tensors must be 2D.");
  
  using Config = typename Config::template KernelSpec<cute::half_t>;

  int M = a.size(0);
  int N = b.size(0);
  int K = a.size(1);

  static constexpr int BlockM = Config::kBlockTiledM;
  static constexpr int BlockN = Config::kBlockTiledN;
  static constexpr int BlockK = Config::kBlockTiledK;

  TORCH_CHECK((M % BlockM) == 0, "M dim must be divisible by ", BlockM);
  TORCH_CHECK((N % BlockN) == 0, "N dim must be divisible by ", BlockN);
  TORCH_CHECK((K % BlockK) == 0, "K dim must be divisible by ", BlockK);

  // cute::print(typename Config::TiledMMA{});

  dim3 block(Config::kThreadNum, 1, 1);
  dim3 grid(
    (M + BlockM - 1) / BlockM, 
    (N + BlockN - 1) / BlockN,
    1
  );
  int shm_size = Config::kShmSize;
  printf("Launching kernel with grid (%d, %d, %d) and block (%d, %d, %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
  printf("Shared memory size: %d\n", shm_size);

  TIME_CUDA_KERNEL(stream,
    gemm_fp16_16_8_8<Config><<<grid, block, shm_size, stream>>>(
      reinterpret_cast<void *>(c.data_ptr()), reinterpret_cast<void *>(a.data_ptr()),
      reinterpret_cast<void *>(b.data_ptr()), M, N, K)
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_fp16_16_8_8", &(run_gemm), py::arg("a"), py::arg("b"), py::arg("c"), "Run a single 16x8x8 MMA operation.");
}