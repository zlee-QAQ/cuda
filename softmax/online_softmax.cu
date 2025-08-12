#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip> // 用于设置输出格式
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat> // 用于 FLT_MAX

// CUDA 错误检查宏
#define CHECK_CUDA_ERROR(err)                                                                                                      \
    if (err != cudaSuccess)                                                                                                        \
    {                                                                                                                              \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << " in file " << __FILE__ << std::endl; \
        exit(EXIT_FAILURE);                                                                                                        \
    }

// =======================================================================
// =================== GPU KERNEL CODE (FROM PREVIOUS STEP) ================
// =======================================================================

// 定义每个线程块的线程数量
#define THREADS 256
// 一个宏，用于将float指针安全地转换为float4指针，以实现向量化访存
#define FLOAT4(val) reinterpret_cast<float4 *>(&(val))[0]

/**
 * @brief 执行Warp级别的在线规约（Online Reduction）。
 */
__device__ void warpreduce_online(float &val_max, float &val_sum)
{
    for (int delta = 16; delta > 0; delta /= 2)
    {
        float partner_max = __shfl_down_sync(0xffffffff, val_max, delta);
        float partner_sum = __shfl_down_sync(0xffffffff, val_sum, delta);

        if (partner_max > val_max)
        {
            val_sum = val_sum * expf(val_max - partner_max) + partner_sum;
            val_max = partner_max;
        }
        else
        {
            val_sum += partner_sum * expf(partner_max - val_max);
        }
    }
}

/**
 * @brief 执行Block级别的在线规约。
 */
template <int thread_per_block = 256>
__device__ void blockreduce_online(float &val_max, float &val_sum)
{
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    constexpr int warp_nums = thread_per_block / 32;

    __shared__ float s_max[warp_nums];
    __shared__ float s_sum[warp_nums];

    warpreduce_online(val_max, val_sum);

    if (threadIdx.x % 32 == 0)
    {
        s_max[warp_id] = val_max;
        s_sum[warp_id] = val_sum;
    }

    __syncthreads();

    // 注意这里的访问，warp_nums可能小于32，所以这里只能用tid 读取，不能用lan_id
    val_max = (tid < warp_nums) ? s_max[tid] : -FLT_MAX;
    val_sum = (tid < warp_nums) ? s_sum[tid] : 0.0f;

    if (warp_id == 0)
    {
        warpreduce_online(val_max, val_sum);
    }
}

/**
 * @brief 计算Softmax的CUDA内核函数（在线版本）。
 */
__global__ void softmax_online(float *in, float *out, int m, int n)
{
    int tid = threadIdx.x;
    int bx = blockIdx.x;

    float *cur_row_in = in + bx * n;
    float *cur_row_out = out + bx * n;

    // --- 阶段1：在线规约 ---
    float thread_max = -INFINITY;
    float thread_sum = 0.0f;

    const int vec_size = 4;
    const int elements_per_thread = vec_size;
    const int elements_per_step = blockDim.x * elements_per_thread;

    for (int offset = tid * elements_per_thread; offset < n; offset += elements_per_step)
    {
        float4 chunk_data = FLOAT4(cur_row_in[offset]);
        float chunk_max = fmaxf(fmaxf(chunk_data.x, chunk_data.y), fmaxf(chunk_data.z, chunk_data.w));

        if (chunk_max > thread_max)
        {
            thread_sum = thread_sum * expf(thread_max - chunk_max);
            thread_max = chunk_max;
        }

        float chunk_sum = expf(chunk_data.x - thread_max) +
                          expf(chunk_data.y - thread_max) +
                          expf(chunk_data.z - thread_max) +
                          expf(chunk_data.w - thread_max);
        thread_sum += chunk_sum;
    }

    blockreduce_online<THREADS>(thread_max, thread_sum);

    __shared__ float final_max_val;
    __shared__ float final_sum_val;
    if (tid == 0)
    {
        final_max_val = thread_max;
        final_sum_val = thread_sum;
    }
    __syncthreads();

    // --- 阶段2：计算并写回 ---
    for (int offset = tid * elements_per_thread; offset < n; offset += elements_per_step)
    {
        float4 data_in = FLOAT4(cur_row_in[offset]);
        float4 data_out;

        data_out.x = expf(data_in.x - final_max_val) / final_sum_val;
        data_out.y = expf(data_in.y - final_max_val) / final_sum_val;
        data_out.z = expf(data_in.z - final_max_val) / final_sum_val;
        data_out.w = expf(data_in.w - final_max_val) / final_sum_val;

        FLOAT4(cur_row_out[offset]) = data_out;
    }
}

// =======================================================================
// =================== CPU & VERIFICATION CODE ===========================
// =======================================================================

/**
 * @brief 在CPU上计算Softmax，用于结果验证。
 * @param in 输入数据指针。
 * @param out 输出数据指针。
 * @param m 行数。
 * @param n 列数。
 */
void softmax_cpu(const std::vector<float> &in, std::vector<float> &out, int m, int n)
{
    // #pragma omp parallel for // 如果CPU核心数多，可以启用OpenMP并行计算
    for (int i = 0; i < m; ++i)
    {
        // 找到当前行的最大值
        float max_val = -FLT_MAX;
        for (int j = 0; j < n; ++j)
        {
            if (in[i * n + j] > max_val)
            {
                max_val = in[i * n + j];
            }
        }

        // 计算指数和
        double sum_exp = 0.0; // 使用double防止求和时精度损失
        for (int j = 0; j < n; ++j)
        {
            sum_exp += expf(in[i * n + j] - max_val);
        }

        // 计算最终的softmax值
        for (int j = 0; j < n; ++j)
        {
            out[i * n + j] = expf(in[i * n + j] - max_val) / sum_exp;
        }
    }
}

/**
 * @brief 验证GPU和CPU的结果是否一致。
 * @param cpu_res CPU计算的参考结果。
 * @param gpu_res GPU计算得到的结果。
 * @param size 数据总长度。
 * @param epsilon 允许的误差范围。
 * @return 如果结果一致则返回true，否则返回false。
 */
bool verify_results(const std::vector<float> &cpu_res, const std::vector<float> &gpu_res, float epsilon = 1e-6f)
{
    for (size_t i = 0; i < cpu_res.size(); ++i)
    {
        if (fabs(cpu_res[i] - gpu_res[i]) > epsilon)
        {
            std::cerr << std::fixed << std::setprecision(16);
            std::cerr << "验证失败于索引 " << i << "！" << std::endl;
            std::cerr << "CPU 结果: " << cpu_res[i] << std::endl;
            std::cerr << "GPU 结果: " << gpu_res[i] << std::endl;
            std::cerr << "差异: " << fabs(cpu_res[i] - gpu_res[i]) << std::endl;
            return false;
        }
    }
    return true;
}

// =======================================================================
// =================== MAIN FUNCTION =====================================
// =======================================================================

int main()
{
    // 1. 定义问题规模
    // const int m = 1024; // 行数 (Batch Size)
    // const int n = 8192; // 列数 (Features)
    const int m = 4;    // 为了方便观察，使用较小规模
    const int n = 1024; // 列数必须是4的倍数

    std::cout << "问题规模: m=" << m << ", n=" << n << std::endl;
    size_t data_size = (size_t)m * n;
    size_t bytes = data_size * sizeof(float);

    // 2. 分配并初始化主机内存
    std::cout << "正在初始化主机数据..." << std::endl;
    std::vector<float> h_in(data_size);
    std::vector<float> h_out_gpu(data_size);
    std::vector<float> h_out_cpu(data_size);

    std::mt19937 rng(12345); // 固定种子以保证每次运行结果相同
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (size_t i = 0; i < data_size; ++i)
    {
        h_in[i] = dist(rng);
    }

    // 3. 分配设备内存
    std::cout << "正在分配设备内存..." << std::endl;
    float *d_in = nullptr, *d_out = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_in, bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out, bytes));

    // 4. 将输入数据从主机拷贝到设备
    std::cout << "正在将数据从主机拷贝到设备..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // 5. 配置并启动CUDA内核
    std::cout << "正在启动CUDA内核..." << std::endl;
    dim3 blockDim(THREADS);
    dim3 gridDim(m); // 每个块处理一行，所以需要m个块

    softmax_online<<<gridDim, blockDim>>>(d_in, d_out, m, n);
    CHECK_CUDA_ERROR(cudaGetLastError());      // 检查内核启动是否有错误
    CHECK_CUDA_ERROR(cudaDeviceSynchronize()); // 等待内核执行完毕

    // 6. 将结果从设备拷回主机
    std::cout << "内核执行完毕，正在将结果拷回主机..." << std::endl;
    CHECK_CUDA_ERROR(cudaMemcpy(h_out_gpu.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // 7. 在CPU上执行计算以供对比
    std::cout << "正在执行CPU版本的Softmax以供对比..." << std::endl;
    softmax_cpu(h_in, h_out_cpu, m, n);

    // 8. 验证结果
    std::cout << "正在验证GPU和CPU结果..." << std::endl;
    if (verify_results(h_out_cpu, h_out_gpu))
    {
        std::cout << "\n\033[32m验证成功！GPU和CPU结果在允许的误差范围内一致。\033[0m" << std::endl;
    }
    else
    {
        std::cout << "\n\033[31m验证失败！GPU和CPU结果不一致。\033[0m" << std::endl;
    }

    // 9. 释放设备内存
    std::cout << "正在释放设备内存..." << std::endl;
    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));

    return 0;
}
