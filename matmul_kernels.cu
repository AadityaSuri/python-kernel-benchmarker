// matmul_kernels.cu
// Example matrix multiplication kernels for benchmarking

__global__ void matmul_naive(const float *a, const float *b, float *res, int N, int M, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        float Pvalue = 0;
        for (int k = 0; k < K; k++) {
            Pvalue += a[row * K + k] * b[k * M + col];
        }
        res[row * M + col] = Pvalue;
    }
}

__global__ void matmul_optimized(const float *a, const float *b, float *res, int N, int M, int K) {
    // Shared memory tiling optimization
    __shared__ float tile_a[16][16];
    __shared__ float tile_b[16][16];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float Pvalue = 0;
    
    // Loop over tiles
    for (int t = 0; t < (K + 15) / 16; t++) {
        // Load tiles into shared memory
        if (row < N && (t * 16 + threadIdx.x) < K)
            tile_a[threadIdx.y][threadIdx.x] = a[row * K + t * 16 + threadIdx.x];
        else
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;
        
        if ((t * 16 + threadIdx.y) < K && col < M)
            tile_b[threadIdx.y][threadIdx.x] = b[(t * 16 + threadIdx.y) * M + col];
        else
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;
        
        __syncthreads();
        
        // Compute partial product
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            Pvalue += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < M) {
        res[row * M + col] = Pvalue;
    }
}

__global__ void matmul_coalesced(const float *a, const float *b, float *res, int N, int M, int K) {
    // Version with better memory coalescing
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < M) {
        float Pvalue = 0;
        
        // Unroll inner loop for better instruction pipelining
        int k = 0;
        for (; k < K - 3; k += 4) {
            Pvalue += a[row * K + k] * b[k * M + col];
            Pvalue += a[row * K + k + 1] * b[(k + 1) * M + col];
            Pvalue += a[row * K + k + 2] * b[(k + 2) * M + col];
            Pvalue += a[row * K + k + 3] * b[(k + 3) * M + col];
        }
        
        // Handle remaining elements
        for (; k < K; k++) {
            Pvalue += a[row * K + k] * b[k * M + col];
        }
        
        res[row * M + col] = Pvalue;
    }
}
