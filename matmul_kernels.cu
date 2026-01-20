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
    // Shared memory tiling with bank conflict avoidance (padding)
    // Simpler than matmul_fast but properly optimized
    __shared__ float tile_a[16][17];  // +1 padding to avoid bank conflicts
    __shared__ float tile_b[16][17];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;

    float Pvalue = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + 15) / 16; t++) {
        // Load tile of A - coalesced access
        int a_col = t * 16 + threadIdx.x;
        if (row < N && a_col < K)
            tile_a[threadIdx.y][threadIdx.x] = a[row * K + a_col];
        else
            tile_a[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile of B - coalesced access
        int b_row = t * 16 + threadIdx.y;
        if (b_row < K && col < M)
            tile_b[threadIdx.y][threadIdx.x] = b[b_row * M + col];
        else
            tile_b[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product - no bank conflicts due to padding
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

// Highly optimized matmul with register tiling and bank conflict avoidance
// Each thread computes a 4x4 block of output (TN x TM per thread)
// Block tile: 64x64, Thread tile: 4x4
#define BN 64   // Block tile size N
#define BM 64   // Block tile size M
#define BK 16   // Block tile size K
#define TN 4    // Thread tile size N (each thread computes TN rows)
#define TM 4    // Thread tile size M (each thread computes TM cols)

__global__ void matmul_fast(const float *a, const float *b, float *res, int N, int M, int K) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ float tile_a[BK][BN + 1];  // +1 padding avoids bank conflicts
    __shared__ float tile_b[BK][BM + 1];

    // Thread indices within block
    const int tx = threadIdx.x;  // 0-15
    const int ty = threadIdx.y;  // 0-15
    const int tid = ty * blockDim.x + tx;  // Linear thread ID (0-255)

    // Each block has 16x16 = 256 threads
    // Each block computes a 64x64 output tile
    // Each thread computes a 4x4 output tile

    // Global starting position for this block's output tile
    const int block_row = blockIdx.y * BN;
    const int block_col = blockIdx.x * BM;

    // Register storage for thread's 4x4 output tile
    float results[TN][TM] = {0.0f};

    // Register storage for loaded A and B values
    float reg_a[TN];
    float reg_b[TM];

    // Loop over K dimension in BK-sized tiles
    for (int t = 0; t < K; t += BK) {
        // Cooperative loading of A tile (64 x 16) using all 256 threads
        // Each thread loads multiple elements
        #pragma unroll
        for (int i = 0; i < (BN * BK) / 256; i++) {
            int load_idx = tid + i * 256;
            int load_row = load_idx / BK;  // Row within tile
            int load_col = load_idx % BK;  // Col within tile
            int global_row = block_row + load_row;
            int global_col = t + load_col;

            if (global_row < N && global_col < K)
                tile_a[load_col][load_row] = a[global_row * K + global_col];
            else
                tile_a[load_col][load_row] = 0.0f;
        }

        // Cooperative loading of B tile (16 x 64) using all 256 threads
        #pragma unroll
        for (int i = 0; i < (BK * BM) / 256; i++) {
            int load_idx = tid + i * 256;
            int load_row = load_idx / BM;  // Row within tile
            int load_col = load_idx % BM;  // Col within tile
            int global_row = t + load_row;
            int global_col = block_col + load_col;

            if (global_row < K && global_col < M)
                tile_b[load_row][load_col] = b[global_row * M + global_col];
            else
                tile_b[load_row][load_col] = 0.0f;
        }

        __syncthreads();

        // Compute partial results for this thread's 4x4 tile
        #pragma unroll
        for (int k = 0; k < BK; k++) {
            // Load A values for this thread's rows into registers
            #pragma unroll
            for (int i = 0; i < TN; i++) {
                reg_a[i] = tile_a[k][ty * TN + i];
            }

            // Load B values for this thread's cols into registers
            #pragma unroll
            for (int j = 0; j < TM; j++) {
                reg_b[j] = tile_b[k][tx * TM + j];
            }

            // Outer product into results
            #pragma unroll
            for (int i = 0; i < TN; i++) {
                #pragma unroll
                for (int j = 0; j < TM; j++) {
                    results[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // Write results to global memory
    #pragma unroll
    for (int i = 0; i < TN; i++) {
        #pragma unroll
        for (int j = 0; j < TM; j++) {
            int global_row = block_row + ty * TN + i;
            int global_col = block_col + tx * TM + j;
            if (global_row < N && global_col < M) {
                res[global_row * M + global_col] = results[i][j];
            }
        }
    }
}
