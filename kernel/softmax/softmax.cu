#include <cuda_runtime.h>
#include <float.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include <torch/extension.h>
#include <torch/types.h>

#define WARP_SIZE 32
#define WARP_NUM 32
#define BLOCK_SIZE 32*32

__device__ float warp_max(float val){
    for(int mask=WARP_SIZE>>1; mask >= 1; mask >>= 1){
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__device__ float warp_sum(float val){
    for(int mask=WARP_SIZE>>1; mask >= 1; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template<bool is_max>
__device__ float block_reduce(float val){
    __shared__ float warp_smem[WARP_NUM];

    float thread_val = 0;
    if (is_max){
        thread_val = warp_max(val);
    } else {
        thread_val = warp_sum(val); 
    }

    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane_id == 0){
        warp_smem[warp_id] = thread_val;
    }

    __syncthreads();

    float init_val = 0;
    if (is_max) {
        init_val = -FLT_MAX;
    } else {
        init_val = 0.0f;
    }

    thread_val = lane_id < WARP_NUM? warp_smem[lane_id] : init_val;
    if (is_max){
        thread_val = warp_max(thread_val);
    } else {
        thread_val = warp_sum(thread_val); 
    }
    return thread_val;
}

__global__ void softmax_kernel(const float* input, float* output, int N) {
    int idx = threadIdx.x;
    int loop_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    float max_val = -FLT_MAX;
    for (int i = 0; i < loop_num; i++){
        int offset = i*blockDim.x + idx;
        float val = offset < N? input[offset] : -FLT_MAX;
        max_val = fmaxf(max_val, val);
    }
    max_val = block_reduce<true>(max_val);

    float sum_val = 0.0f;
    for (int i = 0; i < loop_num; i++){
        int offset = i*blockDim.x + idx;
        float val = offset < N? expf(input[offset] - max_val) : 0.0f;
        sum_val += val;
    }
    sum_val = block_reduce<false>(sum_val);

    for (int i=0; i<loop_num; i++){
        int offset = i*blockDim.x + idx;
        if (offset < N){
            output[offset] = expf(input[offset]-max_val)/sum_val;
        }
    }
}

void run_softmax(const torch::Tensor &input, torch::Tensor &output){
    auto input_size = input.sizes();

    TORCH_CHECK(input_size.size() == 1, "input should be 1D");
    TORCH_CHECK(input.dtype() == torch::kFloat32, "input should be float32");
    TORCH_CHECK(output.dtype() == torch::kFloat32, "output should be float32");
    int N = input_size[0];
    softmax_kernel<<<1, BLOCK_SIZE>>>(reinterpret_cast<float*>(input.data_ptr()), reinterpret_cast<float*>(output.data_ptr()), N);
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, float* output, int N) {
    softmax_kernel<<<1, BLOCK_SIZE>>>(input, output, N);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax", &(run_softmax), py::arg("input"), py::arg("output"), "Run a Softmax operation.");
}