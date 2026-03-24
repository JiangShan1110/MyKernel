#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>
#include <cute/util/print.hpp>

using namespace cute;

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

// template <typename T>
// __global__ void transpose_kernel(void *Bptr, const void *Aptr, int m, int n) {

//   // In software, 128 thread
//   // In hardware, 4 warps, each warp has 32 threads
//   int tid = threadIdx.x;
  
//   // tensor layout
//   Tensor mA = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, n), make_stride(n, Int<1>{})); 
//   Tensor mB = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, m), make_stride(m, Int<1>{}));

//   // Calculate the global thread ID
//   int global_tid = blockIdx.x * blockDim.x + tid;

//   // Each thread will process one element of the output matrix
//   if (global_tid < m * n) {
//     int row = global_tid / n; // Row index in the output matrix
//     int col = global_tid % n; // Column index in the output matrix

//     // Transpose operation: B[col][row] = A[row][col]
//     mB(col, row) = mA(row, col);
//   }
// }

template <typename T, int bm, int bn>
__global__ void transpose_swizzle_kernel(void *Bptr, const void *Aptr, int m, int n) {
    int start_n = blockIdx.x*bn;
    int start_m = blockIdx.y*bm;

    __shared__ float smem[bm][bn];

    for (int offset_m=threadIdx.y; offset_m < bm; offset_m+=blockDim.y) {
        int cur_m = start_m + offset_m;
        if (cur_m >= m) break;
        for (int offset_n=threadIdx.x; offset_n < bn; offset_n+=blockDim.x){
            int cur_n = start_n + offset_n;
            if (cur_n >= n) break;
            smem[offset_m][offset_m^offset_n] = ((T*)Aptr)[cur_m*n + cur_n];
        }
    }

    __syncthreads();

    for (int offset_n=threadIdx.y; offset_n < bn; offset_n+=blockDim.y) {
        int cur_n = start_n + offset_n;
        if (cur_n >= n) break;
        for (int offset_m=threadIdx.x; offset_m < bm; offset_m+=blockDim.x){
            int cur_m = start_m + offset_m;
            if (cur_m >= m) break;
            ((T*)Bptr)[cur_n*m + cur_m] = smem[offset_m][offset_m^offset_n];
        }
    }
  
}

void run_trans(const torch::Tensor &a, torch::Tensor &b){
    auto a_sizes = a.sizes();
    auto b_sizes = b.sizes();

    TORCH_CHECK(a_sizes.size() == 2, "a should be 2D");
    TORCH_CHECK(a_sizes[0] == b_sizes[1] && a_sizes[1] == b_sizes[0], "a and b should be transposes of each other");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "a should be float32");
    TORCH_CHECK(b.dtype() == torch::kFloat32, "b should be float32");

    int M = a_sizes[0];
    int N = a_sizes[1];

    // dim3 blockDim(128);
    // dim3 gridDim((a_sizes[0] * a_sizes[1] + blockDim.x - 1) / blockDim.x);
    // transpose_kernel<cute::half_t><<<gridDim, blockDim>>>(reinterpret_cast<void *>(b.data_ptr()), reinterpret_cast<const void *>(a.data_ptr()), M, N);


    dim3 blockDim(32, 8);
    dim3 gridDim((N + 31) / 32, (M + 7) / 8);
    TIME_CUDA_KERNEL(cudaStreamDefault,
        transpose_swizzle_kernel<float, 32, 32><<<gridDim, blockDim>>>(reinterpret_cast<void *>(b.data_ptr()), reinterpret_cast<const void *>(a.data_ptr()), M, N)
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("transpose", &(run_trans), py::arg("a"), py::arg("b"), "Run a Transpose operation.");
}