#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <limits>

// A helper macro to check for CUDA errors
#define CHECK_CUDA(call)                                                                             \
    do                                                                                               \
    {                                                                                                \
        cudaError_t err = call;                                                                      \
        if (err != cudaSuccess)                                                                      \
        {                                                                                            \
            fprintf(stderr, "CUDA Error: %s:%d, %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                      \
        }                                                                                            \
    } while (0)

#define THREADS 256
#define FLOAT4(val) reinterpret_cast<float4 *>(&(val))[0]

__device__ float warpreducesum(float val)
{
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

template <int thread_per_block = 256>
__device__ float blockreducesum(float val)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    constexpr int warp_nums = thread_per_block / 32;
    __shared__ float sdata[warp_nums];

    val = warpreducesum(val); // Each warp reduces its sum

    // Only the first thread in each warp writes its sum to shared memory
    if (threadIdx.x % 32 == 0)
    {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // The first warp now reduces the partial sums from shared memory
    val = (tid < warp_nums) ? sdata[tid] : 0.0f;
    if (warp_id == 0)
    {
        val = warpreducesum(val);
    }
    // The first thread broadcasts the final sum
    if (tid == 0)
        sdata[0] = val;
    __syncthreads();
    return sdata[0];
}

__device__ float warpreducemax(float val)
{
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
    return val;
}

template <int thread_per_block = 256>
__device__ float blockreducemax(float val)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    constexpr int warp_nums = thread_per_block / 32;
    __shared__ float sdata[warp_nums];

    val = warpreducemax(val); // Each warp reduces its max

    // Only the first thread in each warp writes its max to shared memory
    if (threadIdx.x % 32 == 0)
    {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // The first warp now reduces the partial max values from shared memory
    val = (tid < warp_nums) ? sdata[tid] : -INFINITY;
    if (warp_id == 0)
    {
        val = warpreducemax(val);
    }
    // The first thread broadcasts the final max
    if (tid == 0)
        sdata[0] = val;
    __syncthreads();
    return sdata[0];
}
// 每个threadblock 负责一行
__global__ void softmax(float *in, float *out, int m, int n)
{
    int tid = threadIdx.x;
    int bx = blockIdx.x;

    float *cur_row = in + blockIdx.x * n;

    // reduce max
    float thread_max = -INFINITY;
    int thread_pos = tid * 4;
    int elements_per_step = blockDim.x * 4;
    for (int offset = 0; offset < n; offset += elements_per_step)
    {
        float4 tmp = FLOAT4(cur_row[offset + thread_pos]);
        thread_max = fmaxf(thread_max, fmaxf(tmp.x, tmp.w));
        thread_max = fmaxf(thread_max, fmaxf(tmp.y, tmp.z));
    }

    float max_val = blockreducemax(thread_max);
    float thread_sum = 0.0f;
    for (int offset = 0; offset < n; offset += elements_per_step)
    {
        float4 tmp = FLOAT4(cur_row[offset + thread_pos]);
        thread_sum += expf(tmp.w - max_val);
        thread_sum += expf(tmp.x - max_val);
        thread_sum += expf(tmp.y - max_val);
        thread_sum += expf(tmp.z - max_val);
    }

    float exp_sum = blockreducesum(thread_sum);

    float *out_row = out + +blockIdx.x * n;

    for (int offset = 0; offset < n; offset += elements_per_step)
    {
        float4 tmp = FLOAT4(cur_row[offset + thread_pos]);
        tmp.w = expf(tmp.w - max_val) / exp_sum;
        tmp.x = expf(tmp.x - max_val) / exp_sum;
        tmp.y = expf(tmp.y - max_val) / exp_sum;
        tmp.z = expf(tmp.z - max_val) / exp_sum;
        FLOAT4(out_row[offset + thread_pos]) = tmp;
    }
}

// 用于验证的 CPU softmax 实现
void cpu_softmax(const std::vector<float> &in, std::vector<float> &out, int m, int n)
{
    for (int i = 0; i < m; ++i)
    {
        const float *row_in = &in[i * n];
        float *row_out = &out[i * n];

        // 1. 找到该行的最大值以保证数值稳定性
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < n; ++j)
        {
            if (row_in[j] > max_val)
            {
                max_val = row_in[j];
            }
        }

        // 2. 计算指数和
        double sum = 0.0; // 使用 double 以获得更高精度
        for (int j = 0; j < n; ++j)
        {
            sum += expf(row_in[j] - max_val);
        }

        // 3. 计算 Softmax
        for (int j = 0; j < n; ++j)
        {
            row_out[j] = expf(row_in[j] - max_val) / sum;
        }
    }
}

int main()
{
    // --- 测试参数 ---
    int m = 256;  // 矩阵行数 (对应线程块数)
    int n = 4096; // 矩阵列数
    printf("矩阵尺寸: %d x %d\n", m, n);

    // --- 主机内存分配和初始化 ---
    size_t bytes = m * n * sizeof(float);
    std::vector<float> h_in(m * n);
    std::vector<float> h_out_gpu(m * n);
    std::vector<float> h_out_cpu(m * n);

    // 用一些值初始化输入数据
    for (int i = 0; i < m * n; ++i)
    {
        h_in[i] = static_cast<float>((i % 100) - 50); // 示例数据范围
    }

    // --- 设备内存分配 ---
    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));

    // --- 数据传输：主机 -> 设备 ---
    printf("正在从主机向设备复制输入数据...\n");
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // --- 启动核函数 ---
    dim3 gridDim(m); // 每行一个线程块
    dim3 blockDim(THREADS);
    printf("正在启动 softmax 核函数...\n");
    softmax<<<gridDim, blockDim>>>(d_in, d_out, m, n);
    CHECK_CUDA(cudaGetLastError());      // 检查启动错误
    CHECK_CUDA(cudaDeviceSynchronize()); // 等待核函数执行完毕

    // --- 数据传输：设备 -> 主机 ---
    printf("正在从设备向主机复制输出数据...\n");
    CHECK_CUDA(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // --- CPU 验证 ---
    printf("正在 CPU 上计算参考解...\n");
    cpu_softmax(h_in, h_out_cpu, m, n);

    // --- 结果比对 ---
    bool success = true;
    float epsilon = 1e-10f; // 用于浮点数比较的容差
    for (int i = 0; i < m * n; ++i)
    {
        if (fabs(h_out_gpu[i] - h_out_cpu[i]) > epsilon)
        {
            printf("索引 %d 处发现不匹配! GPU: %.18f, CPU: %.18f\n", i, h_out_gpu[i], h_out_cpu[i]);
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("✅ 测试通过！\n");
    }
    else
    {
        printf("❌ 测试失败！\n");
    }

    // --- 清理工作 ---
    printf("正在清理 GPU 内存...\n");
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}
