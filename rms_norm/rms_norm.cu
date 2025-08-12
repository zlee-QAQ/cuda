#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>

// CUDA 运行时头文件
#include <cuda_runtime.h>

// 用于检查 CUDA API 调用的宏
#define CUDA_CHECK(call)                                                     \
    do                                                                       \
    {                                                                        \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess)                                              \
        {                                                                    \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define FLOAT4(val) reinterpret_cast<float4 *>(&(val))[0]
#define THREAD_PER_BLOCK 256

// ============================================================================
// <<< KERNEL CODE (FROM PROMPT) >>>
// ============================================================================

__device__ void warpsum(float &thread_sum)
{
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, 16);
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, 8);
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, 4);
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, 2);
    thread_sum += __shfl_down_sync(0xffffffff, thread_sum, 1);
}

__device__ void blockreduce_sum(float &thread_sum)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    constexpr int warp_nums = THREAD_PER_BLOCK / 32;
    __shared__ float s_data[warp_nums];

    warpsum(thread_sum);
    if (lane_id == 0)
        s_data[warp_id] = thread_sum;

    __syncthreads();

    thread_sum = tid < warp_nums ? s_data[tid] : 0.0f;
    if (warp_id == 0)
        warpsum(thread_sum);
}

__global__ void rmsnorm(float *in, float *out, float e, float *w, int m, int n)
{
    int tid = threadIdx.x;
    float *cur_row = in + blockIdx.x * n;

    float thread_sum = 0.0f;
    int elements_per_step = 4 * THREAD_PER_BLOCK;
    for (int offset = tid * 4; offset < n; offset += elements_per_step)
    {
        float4 tmp = FLOAT4(cur_row[offset]);
        thread_sum += tmp.x * tmp.x + tmp.y * tmp.y + tmp.z * tmp.z + tmp.w * tmp.w;
    }

    blockreduce_sum(thread_sum);

    __shared__ float sum;
    if (tid == 0)
        sum = thread_sum;
    __syncthreads();

    sum = rsqrtf(sum / n + e);

    float *out_row = out + blockIdx.x * n;
    // It's slightly more efficient to read the weight once before the loop
    float w0 = w[blockIdx.x];
    for (int offset = tid * 4; offset < n; offset += elements_per_step)
    {
        float4 tmp = FLOAT4(cur_row[offset]);
        tmp.x = tmp.x * sum * w0;
        tmp.y = tmp.y * sum * w0;
        tmp.z = tmp.z * sum * w0;
        tmp.w = tmp.w * sum * w0;

        FLOAT4(out_row[offset]) = tmp;
    }
}

// ============================================================================
// <<< TEST HARNESS AND CPU VERIFICATION >>>
// ============================================================================

// CPU implementation of RMSNorm for verification
void cpu_rmsnorm(const float *in, float *out, float e, const float *w, int m, int n)
{
    // Process one row at a time (m is the batch size)
    for (int i = 0; i < m; ++i)
    {
        const float *current_in_row = in + i * n;
        float *current_out_row = out + i * n;

        // 1. Calculate the sum of squares for the row
        double sum_sq = 0.0; // Use double for better precision in sum
        for (int j = 0; j < n; ++j)
        {
            sum_sq += static_cast<double>(current_in_row[j]) * current_in_row[j];
        }

        // 2. Calculate the reciprocal of the root mean square (1 / RMS)
        sum_sq /= n;
        sum_sq += e;
        float inv_rms = 1.0f / sqrtf(static_cast<float>(sum_sq));

        // 3. Normalize the row and apply the weight
        float weight = w[i];
        for (int j = 0; j < n; ++j)
        {
            current_out_row[j] = current_in_row[j] * inv_rms * weight;
        }
    }
}

// Function to compare CPU and GPU results with a tolerance
void verify_results(const std::vector<float> &cpu_out, const std::vector<float> &gpu_out, float tolerance = 1e-4f)
{
    assert(cpu_out.size() == gpu_out.size());
    size_t size = cpu_out.size();
    bool passed = true;

    for (size_t i = 0; i < size; ++i)
    {
        if (fabsf(cpu_out[i] - gpu_out[i]) > tolerance)
        {
            std::cerr << "Mismatch at index " << i << ": CPU = " << cpu_out[i]
                      << ", GPU = " << gpu_out[i] << ", Diff = " << fabsf(cpu_out[i] - gpu_out[i]) << std::endl;
            passed = false;
            // Stop after finding the first error for brevity
            break;
        }
    }

    if (passed)
    {
        std::cout << "✅ Test Passed! CPU and GPU results match." << std::endl;
    }
    else
    {
        std::cout << "❌ Test Failed! CPU and GPU results differ." << std::endl;
    }
}

// Main function to drive the test
int main()
{
    // --- Test Parameters ---
    const int m = 16;   // Batch size (number of rows)
    const int n = 1024; // Feature size (must be a multiple of 4)
    const float epsilon = 1e-5f;

    std::cout << "Running RMSNorm test with:" << std::endl;
    std::cout << "  Batch Size (m): " << m << std::endl;
    std::cout << "  Feature Size (n): " << n << std::endl;

    // --- Host Data Initialization ---
    std::vector<float> h_in(m * n);
    std::vector<float> h_w(m);
    std::vector<float> h_gpu_out(m * n);
    std::vector<float> h_cpu_out(m * n);

    // Initialize input data and weights with some values
    for (int i = 0; i < m * n; ++i)
    {
        h_in[i] = static_cast<float>(i % 128) * 0.1f - 6.4f;
    }
    for (int i = 0; i < m; ++i)
    {
        h_w[i] = 1.0f + static_cast<float>(i) * 0.1f;
    }

    // --- GPU Execution ---
    float *d_in, *d_out, *d_w;
    CUDA_CHECK(cudaMalloc(&d_in, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, m * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w, m * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), m * sizeof(float), cudaMemcpyHostToDevice));

    dim3 blockDim(THREAD_PER_BLOCK);
    dim3 gridDim(m); // One block per row

    rmsnorm<<<gridDim, blockDim>>>(d_in, d_out, epsilon, d_w, m, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu_out.data(), d_out, m * n * sizeof(float), cudaMemcpyDeviceToHost));

    // --- CPU Execution ---
    cpu_rmsnorm(h_in.data(), h_cpu_out.data(), epsilon, h_w.data(), m, n);

    // --- Verification ---
    verify_results(h_cpu_out, h_gpu_out);

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_w));

    return 0;
}
