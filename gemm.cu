#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cuda_runtime.h>

// 用于检查 CUDA 错误的宏
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        std::cerr << "CUDA 错误 = " << static_cast<unsigned int>(result) << " (" << cudaGetErrorString(result) << ") 在 " << file << ":" << line << " '" << func << "' \n";
        // 在出现错误时重置设备并退出
        cudaDeviceReset();
        exit(99);
    }
}

#define OFFSET(row, col, width) ((row) * (width)) + (col)
#define FLOAT4(val) reinterpret_cast<float4 *>(&(val))[0]

template<int tile_size_m,
         int tile_size_n,
         int tile_size_k,
         int thread_size_m,
         int thread_size_n>
__global__ void gemm(float *a, float *b, float *c,
                     int m, int n, int k)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tid = ty * blockDim.x + tx;
    int threads_per_block = blockDim.x * blockDim.y;

    __shared__ float as[2][tile_size_m][tile_size_k];
    __shared__ float bs[2][tile_size_k][tile_size_n];

    float accum[thread_size_m][thread_size_n] = {0.0f};
    float frag_a[thread_size_m];
    float frag_b[thread_size_n];

    // 每个线程负责 4 个连续元素
    const int threads_per_row_a = tile_size_k / 4;
    const int threads_per_row_b = tile_size_n / 4;

    const int row_stride_a = threads_per_block / threads_per_row_a;
    const int row_stride_b = threads_per_block / threads_per_row_b;

    const int thread_start_row_a = tid / threads_per_row_a;
    const int thread_start_col_a = (tid % threads_per_row_a) * 4;

    const int thread_start_row_b = tid / threads_per_row_b;
    const int thread_start_col_b = (tid % threads_per_row_b) * 4;

    /*-------------------------------------------------
     * 阶段 1：预加载第 0 个 tile
     *------------------------------------------------*/
    int write_buf = 0;

    // 加载 A tile
    for (int row_offset = 0; row_offset < tile_size_m; row_offset += row_stride_a)
    {
        int sm_row = thread_start_row_a + row_offset;
        int gm_row = by * tile_size_m + sm_row;
        int gm_col = thread_start_col_a;
        FLOAT4(as[write_buf][sm_row][thread_start_col_a]) =
                    FLOAT4(a[OFFSET(gm_row, gm_col, k)]);
    }

    // 加载 B tile
    for (int row_offset = 0; row_offset < tile_size_k; row_offset += row_stride_b)
    {
        int sm_row = thread_start_row_b + row_offset;
        int gm_row = sm_row;                 // B 行对应 K 维
        int gm_col = bx * tile_size_n + thread_start_col_b;
        FLOAT4(bs[write_buf][sm_row][thread_start_col_b]) =
                FLOAT4(b[OFFSET(gm_row, gm_col, n)]);
    }
    __syncthreads();

    /*-------------------------------------------------
     * 阶段 2：主循环，双缓冲
     *------------------------------------------------*/
    for (int tile_k_idx = 0; tile_k_idx < k; tile_k_idx += tile_size_k)
    {
        int read_buf  = write_buf;
        write_buf     = 1 - read_buf;

        int next_tile_k_start = tile_k_idx + tile_size_k;

        // 预取下一个 tile
        if (next_tile_k_start < k)
        {
            // A tile
            for (int row_offset = 0; row_offset < tile_size_m; row_offset += row_stride_a)
            {
                int sm_row = thread_start_row_a + row_offset;
                int gm_row = by * tile_size_m + sm_row;
                int gm_col = next_tile_k_start + thread_start_col_a;
                FLOAT4(as[write_buf][sm_row][thread_start_col_a]) = FLOAT4(a[OFFSET(gm_row, gm_col, k)]);
            }

            // B tile
            for (int row_offset = 0; row_offset < tile_size_k; row_offset += row_stride_b)
            {
                int sm_row = thread_start_row_b + row_offset;
                    int gm_row = next_tile_k_start + sm_row;
                    int gm_col = bx * tile_size_n + thread_start_col_b;
                        FLOAT4(bs[write_buf][sm_row][thread_start_col_b]) =
                            FLOAT4(b[OFFSET(gm_row, gm_col, n)]);
            }
        }

        /* 计算当前 tile */
        for (int k_idx = 0; k_idx < tile_size_k; ++k_idx)
        {
            // A 片段
            for (int i = 0; i < thread_size_m; ++i)
                frag_a[i] = as[read_buf][ty * thread_size_m + i][k_idx];

            // B 片段（矢量加载）
            for (int j = 0; j < thread_size_n; j += 4)
                FLOAT4(frag_b[j]) = FLOAT4(bs[read_buf][k_idx][tx * thread_size_n + j]);

            // 乘加
            for (int i = 0; i < thread_size_m; ++i)
                for (int j = 0; j < thread_size_n; ++j)
                    accum[i][j] += frag_a[i] * frag_b[j];
        }
        __syncthreads();
    }

    /*-------------------------------------------------
     * 阶段 3：写回 C
     *------------------------------------------------*/
    int c_row_base = by * tile_size_m + ty * thread_size_m;
    int c_col_base = bx * tile_size_n + tx * thread_size_n;

    for (int i = 0; i < thread_size_m; ++i)
    {
        for (int j = 0; j < thread_size_n; ++j)
        {
            int row = c_row_base + i;
            int col = c_col_base + j;
            c[OFFSET(row, col, n)] = accum[i][j];
        }
    }
}



// 在 CPU 上进行简单的串行 GEMM 实现，用于验证结果
void cpu_gemm(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c, int M, int N, int K)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            float sum = 0.0f;
            for (int l = 0; l < K; ++l)
            {
                sum += a[OFFSET(i, l, K)] * b[OFFSET(l, j, N)];
            }
            c[OFFSET(i, j, N)] = sum;
        }
    }
}

// 比较两个矩阵并检查结果是否正确
void verify_result(const std::vector<float> &gpu_result, const std::vector<float> &cpu_result, int M, int N)
{
    const float epsilon = 1e-2f; // 针对大规模浮点累加，使用相对容差可能更稳健
    bool correct = true;
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int index = OFFSET(i, j, N);
            // 使用相对误差和绝对误差相结合的方式进行比较
            if (std::abs(gpu_result[index] - cpu_result[index]) > epsilon * std::max(std::abs(cpu_result[index]), std::abs(gpu_result[index])) && std::abs(gpu_result[index] - cpu_result[index]) > epsilon)
            {
                std::cerr << "验证失败于 (" << i << ", " << j << ")! "
                          << "GPU 结果: " << gpu_result[index]
                          << ", CPU 结果: " << cpu_result[index] << std::endl;
                correct = false;
                goto verification_end; // 在发现第一个错误时提前退出循环
            }
        }
    }

verification_end:
    if (correct)
    {
        std::cout << "验证通过！" << std::endl;
    }
    else
    {
        std::cout << "验证失败！" << std::endl;
    }
}

int main()
{
    // --- 问题定义 ---
    const int M = 1024;
    const int N = 512;
    const int K = 512;

    // --- 核函数配置 ---
    const int tile_size_m = 128;
    const int tile_size_n = 128;
    const int tile_size_k = 8;
    const int thread_size_m = 16;
    const int thread_size_n = 16;

    const int THREADS_X = tile_size_n / thread_size_n; // 128 / 8 = 16
    const int THREADS_Y = tile_size_m / thread_size_m; // 128 / 8 = 16

    std::cout << "矩阵维度: C(" << M << "x" << N << ") = A(" << M << "x" << K << ") * B(" << K << "x" << N << ")" << std::endl;
    std::cout << "核函数配置: Tile(" << tile_size_m << "x" << tile_size_n << "x" << tile_size_k << "), ThreadWork(" << thread_size_m << "x" << thread_size_n << ")" << std::endl;

    // --- 主机内存配置与初始化 ---
    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    std::vector<float> h_c_gpu(M * N, 0.0f);
    std::vector<float> h_c_cpu(M * N, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < M * K; ++i)
        h_a[i] = dis(gen);
    for (int i = 0; i < K * N; ++i)
        h_b[i] = dis(gen);

    // --- 设备内存配置 ---
    float *d_a, *d_b, *d_c;
    checkCudaErrors(cudaMalloc(&d_a, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_b, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_c, M * N * sizeof(float)));

    // --- 将数据从主机复制到设备 ---
    checkCudaErrors(cudaMemcpy(d_a, h_a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_b, h_b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));

    // --- 核函数启动 ---
    dim3 blockDim(THREADS_X, THREADS_Y);
    dim3 gridDim((N + tile_size_n - 1) / tile_size_n, (M + tile_size_m - 1) / tile_size_m);

    std::cout << "启动核函数，网格维度(" << gridDim.x << ", " << gridDim.y << ")，区块维度(" << blockDim.x << ", " << blockDim.y << ")" << std::endl;

    gemm<tile_size_m, tile_size_n, tile_size_k, thread_size_m, thread_size_n><<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // --- 将结果从设备复制到主机 ---
    checkCudaErrors(cudaMemcpy(h_c_gpu.data(), d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // --- 验证 ---
    std::cout << "在 CPU 上计算参考结果..." << std::endl;
    cpu_gemm(h_a, h_b, h_c_cpu, M, N, K);

    std::cout << "验证 GPU 结果与 CPU 结果..." << std::endl;
    verify_result(h_c_gpu, h_c_cpu, M, N);

    // --- 清理 ---
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    return 0;
}
