#include <iostream>
#include <vector>
#include <random>
#include <cfloat> // For FLT_MAX

// CUDA 运行时API
#include <cuda_runtime.h>

// Helper macro for checking CUDA API calls
#define CUDA_CHECK(call)                                          \
    do                                                            \
    {                                                             \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n",          \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

/**
 * @brief CUDA核函数，用于在4D矩阵的第二个维度上执行reduce max操作
 *
 * @param input  指向输入4D矩阵 (256, 16, 128, 128) 的设备指针
 * @param output 指向输出3D矩阵 (256, 128, 128) 的设备指针
 * @param D1     维度1的大小 (256)
 * @param D2     维度2的大小 (16, a.k.a. reduce_dim)
 * @param D3     维度3的大小 (128)
 * @param D4     维度4的大小 (128)
 */
__global__ void reduceMaxKernel(const float *input, float *output, int D1, int D2, int D3, int D4)
{
    // 计算此线程将要处理的输出元素在 (D1, D3, D4) 空间中的全局索引
    // 我们将输出的三维空间 (D1, D3, D4) 映射到CUDA的执行网格上
    int idx_d1 = blockIdx.z;
    int idx_d3 = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_d4 = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查，确保线程索引不会超出输出矩阵的维度
    if (idx_d1 < D1 && idx_d3 < D3 && idx_d4 < D4)
    {
        // 初始化最大值为一个非常小的数
        float max_val = -FLT_MAX;

        // 沿着要规约的维度 (D2) 进行循环
        for (int l = 0; l < D2; ++l)
        {
            // 计算当前元素在输入4D矩阵中的线性索引 (row-major order)
            // index = i*dim2*dim3*dim4 + l*dim3*dim4 + j*dim4 + k
            long long input_idx = (long long)idx_d1 * D2 * D3 * D4 +
                                  (long long)l * D3 * D4 +
                                  (long long)idx_d3 * D4 +
                                  idx_d4;

            // 更新最大值
            max_val = max(max_val, input[input_idx]);
        }

        // 计算输出元素在输出3D矩阵中的线性索引
        long long output_idx = (long long)idx_d1 * D3 * D4 +
                               (long long)idx_d3 * D4 +
                               idx_d4;

        // 将找到的最大值写入输出矩阵
        output[output_idx] = max_val;
    }
}

/**
 * @brief 在CPU上执行reduce max以进行验证
 */
void verifyOnCPU(const std::vector<float> &h_input, std::vector<float> &h_output_cpu, int D1, int D2, int D3, int D4)
{
    for (int i = 0; i < D1; ++i)
    {
        for (int j = 0; j < D3; ++j)
        {
            for (int k = 0; k < D4; ++k)
            {
                float max_val = -FLT_MAX;
                for (int l = 0; l < D2; ++l)
                {
                    long long input_idx = (long long)i * D2 * D3 * D4 +
                                          (long long)l * D3 * D4 +
                                          (long long)j * D4 +
                                          k;
                    if (h_input[input_idx] > max_val)
                    {
                        max_val = h_input[input_idx];
                    }
                }
                long long output_idx = (long long)i * D3 * D4 +
                                       (long long)j * D4 +
                                       k;
                h_output_cpu[output_idx] = max_val;
            }
        }
    }
}

/**
 * @brief 使用float4向量化访存实现的CUDA核函数，用于在4D矩阵的第二个维度上执行reduce max操作。
 * 每个线程负责计算输出中D4维度上连续的4个元素。
 *
 * @param input  指向输入4D矩阵 (256, 16, 128, 128) 的设备指针
 * @param output 指向输出3D矩阵 (256, 128, 128) 的设备指针
 * @param D1     维度1的大小 (256)
 * @param D2     维度2的大小 (16, a.k.a. reduce_dim)
 * @param D3     维度3的大小 (128)
 * @param D4     维度4的大小 (128)
 */
__global__ void reduceMaxKernel_Vectorized(const float *input, float *output, int D1, int D2, int D3, int D4)
{
    // 1. 线程到数据的映射
    // 每个线程将处理D4维度上的4个连续元素。
    // threadIdx.x 计算的是float4向量的索引，所以实际的float索引需要乘以4。
    int idx_d4_start = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int idx_d3 = blockIdx.y * blockDim.y + threadIdx.y;
    int idx_d1 = blockIdx.z;

    // 边界检查：确保线程处理的起始位置在有效范围内。
    if (idx_d1 < D1 && idx_d3 < D3 && idx_d4_start < D4)
    {
        // 2. 初始化
        // 为要处理的4个元素分别初始化最大值。
        float max_vals[4] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

        // 3. 循环规约
        // 沿需要规约的维度 D2 进行循环。
        for (int l = 0; l < D2; ++l)
        {
            // 计算当前 (d1, l, d3) "行" 的起始线性索引
            long long row_base_idx = ((long long)idx_d1 * D2 * D3 * D4) +
                                     ((long long)l * D3 * D4) +
                                     ((long long)idx_d3 * D4);

            // 构造指向当前线程所需float4数据的指针
            const float4 *vec_ptr = reinterpret_cast<const float4 *>(input + row_base_idx + idx_d4_start);

            // 向量化加载：一次性读取4个float值
            float4 current_vals = *vec_ptr;

            // 并行更新4个最大值
            max_vals[0] = fmaxf(max_vals[0], current_vals.x);
            max_vals[1] = fmaxf(max_vals[1], current_vals.y);
            max_vals[2] = fmaxf(max_vals[2], current_vals.z);
            max_vals[3] = fmaxf(max_vals[3], current_vals.w);
        }

        // 4. 写回结果
        // 计算输出位置的起始线性索引
        long long output_base_idx = ((long long)idx_d1 * D3 * D4) +
                                    ((long long)idx_d3 * D4);

        // 构造指向输出位置的指针
        float4 *output_vec_ptr = reinterpret_cast<float4 *>(output + output_base_idx + idx_d4_start);

        // 向量化写入：一次性将4个结果写入全局内存
        *output_vec_ptr = make_float4(max_vals[0], max_vals[1], max_vals[2], max_vals[3]);
    }
}
int main()
{
    // 定义矩阵维度
    const int D1 = 256;
    const int D2 = 16; // 这是我们要规约的维度
    const int D3 = 128;
    const int D4 = 128;

    // 计算输入和输出矩阵的元素总数
    long long input_size = (long long)D1 * D2 * D3 * D4;
    long long output_size = (long long)D1 * D3 * D4;

    // 计算内存大小（以字节为单位）
    size_t input_bytes = input_size * sizeof(float);
    size_t output_bytes = output_size * sizeof(float);

    std::cout << "Input matrix size: " << D1 << "x" << D2 << "x" << D3 << "x" << D4 << std::endl;
    std::cout << "Output matrix size: " << D1 << "x" << D3 << "x" << D4 << std::endl;

    // 在主机端 (CPU) 分配内存
    std::vector<float> h_input(input_size);
    std::vector<float> h_output_gpu(output_size);
    std::vector<float> h_output_cpu(output_size);

    // 初始化输入数据
    std::cout << "Initializing host data..." << std::endl;
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(0.0f, 100.0f);
    for (long long i = 0; i < input_size; ++i)
    {
        h_input[i] = distribution(generator);
    }

    // 在设备端 (GPU) 分配内存
    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    // 将输入数据从主机复制到设备
    std::cout << "Copying data from host to device..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));

    // 定义核函数启动的线程块和网格维度
    // 我们为输出矩阵的 (D3, D4) 平面定义一个2D线程块
    dim3 threadsPerBlock(16, 16);
    // 我们定义一个3D网格来覆盖 (D3, D4, D1) 维度
    dim3 numBlocks((D3 + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (D4 + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   D1);

    std::cout << "Launching CUDA kernel..." << std::endl;

    // 启动核函数
    reduceMaxKernel_Vectorized<<<numBlocks, threadsPerBlock>>>(d_input, d_output, D1, D2, D3, D4);

    // 检查核函数启动是否有错误
    CUDA_CHECK(cudaGetLastError());
    // 同步设备以确保核函数执行完毕
    CUDA_CHECK(cudaDeviceSynchronize());

    // 将结果从设备复制回主机
    std::cout << "Copying result from device to host..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_output_gpu.data(), d_output, output_bytes, cudaMemcpyDeviceToHost));

    // 在CPU上执行计算以进行验证
    std::cout << "Verifying result on CPU..." << std::endl;
    verifyOnCPU(h_input, h_output_cpu, D1, D2, D3, D4);

    // 比较GPU和CPU的结果
    bool success = true;
    for (long long i = 0; i < output_size; ++i)
    {
        if (fabs(h_output_gpu[i] - h_output_cpu[i]) > 1e-5)
        {
            std::cerr << "Verification failed at index " << i << "! ";
            std::cerr << "GPU result: " << h_output_gpu[i] << ", CPU result: " << h_output_cpu[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success)
    {
        std::cout << "Verification successful!" << std::endl;
    }
    else
    {
        std::cout << "Verification failed!" << std::endl;
    }

    // 释放设备内存
    std::cout << "Freeing device memory..." << std::endl;
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
