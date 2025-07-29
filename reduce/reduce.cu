__device__ void reduce_in_warp(volatile float *cache, int tid)
{
    // 假设warp大小为32
    cache[tid] += cache[tid + 32];
    cache[tid] += cache[tid + 16];
    cache[tid] += cache[tid + 8];
    cache[tid] += cache[tid + 4];
    cache[tid] += cache[tid + 2];
    cache[tid] += cache[tid + 1];
}

__global__ void reduce_baseline(float *d_in, float *d_out)
{
    __shared__ float sdata[THREAD_PER_BLOCK];

    unsigned int tid = threadIdx.x;
    unsigned int tx = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    sdata[tid] = d_in[tx] + d_in[tx + blockDim.x];
    __syncthreads();

// 优化bank conflict
#pragma unroll
    for (int i = blockDim.x / 2; i > 32; i >>= 1)
    {
        if (tid < i)
        {
            sdata[tid] += sdata[tid + i];
        }

        __syncthreads();
    }

    if (tid < 32)
        reduce_in_warp(sdata, tid);
    if (tid == 0)
        d_out[blockIdx.x] = sdata[tid];
}
