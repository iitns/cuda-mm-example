#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <thread>
#include <algorithm>
#include <cstring>
using namespace std;

#define N 2048
#define TILE_SIZE 16

// ======================
// CPU (single thread)
// ======================
void cpu_mm_naive(const int* A, const int* B, int* C)
{
    for (int i = 0; i < N; i++) {         // i: row index of output C
        for (int j = 0; j < N; j++) {     // j: column index of output C
            int sum = 0;                   // accumulator for dot product C[i][j]
            for (int k = 0; k < N; k++) { // k: inner dimension, C[i][j] = sum_k A[i][k]*B[k][j]
                // Access pattern (row-major): C[i][j] += A[i][k] * B[k][j]
                //   A[i*N+k] = A[i][k] = element in row i, column k
                //   B[k*N+j] = B[k][j] = element in row k, column j
                //
                //     A (row i)     *     B (col j)      => one term for C[i][j]
                //   [ ... A[i][k] ... ]   [ B[0][j] ]
                //                         [   ...   ]
                //                         [ B[k][j] ]  ← same k
                //                         [   ...   ]
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}


// ======================
// CPU (multi thread)
// ======================
void cpu_mm_naive_worker(const int* A, const int* B, int* C, int row_begin, int row_end)
{
    for (int i = row_begin; i < row_end; i++) {  // i: rows assigned to this thread
        for (int j = 0; j < N; j++) {
            int sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

unsigned int cpu_mm_naive_mt(const int* A, const int* B, int* C)
{
    unsigned int num_threads = thread::hardware_concurrency();
    if (num_threads == 0) {
        num_threads = 1;
    }

    num_threads = min<unsigned int>(num_threads, static_cast<unsigned int>(N));

    vector<thread> workers;
    workers.reserve(num_threads);

    int rows_per_thread = N / static_cast<int>(num_threads);  // base row count per thread
    int remainder = N % static_cast<int>(num_threads);        // extra rows to distribute

    int row_start = 0;
    for (unsigned int t = 0; t < num_threads; ++t) {
        int extra = (t < static_cast<unsigned int>(remainder)) ? 1 : 0;  // give one extra row to first 'remainder' threads
        int row_end = row_start + rows_per_thread + extra;

        workers.emplace_back(cpu_mm_naive_worker, A, B, C, row_start, row_end);
        row_start = row_end;
    }

    for (auto& th : workers) {
        th.join();
    }

    return num_threads;
}

// ======================
// GPU (naive)
// ======================
__global__ void mm_naive(int* A, int* B, int* C)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // global row index for this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // global column index for this thread

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


// ======================
// GPU (Tiled MM (assuming N % TILE_SIZE == 0))
// ======================
__global__ void matrixMulTiled(const int* A, const int* B, int* C)
{
    __shared__ int sA[TILE_SIZE][TILE_SIZE];  // tile of A in shared memory
    __shared__ int sB[TILE_SIZE][TILE_SIZE];  // tile of B in shared memory

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;   // global row this thread is responsible for
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // global column this thread is responsible for

    int sum = 0;                          // partial sum for C[row][col]
    int numtiles = N / TILE_SIZE;         // number of tiles along inner dimension

    for (int t = 0; t < numtiles; t++) {
        sA[threadIdx.y][threadIdx.x] =
            A[row * N + (t * TILE_SIZE + threadIdx.x)];   // load A tile: row 'row', columns t*TILE_SIZE..t*TILE_SIZE+TILE_SIZE-1

        sB[threadIdx.y][threadIdx.x] =
            B[(t * TILE_SIZE + threadIdx.y) * N + col];   // load B tile: rows t*TILE_SIZE.., column 'col'

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];  // dot product within tile
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}


int main()
{
    int* h_A = new int[N * N];   // host: input matrix A
    int* h_B = new int[N * N];   // host: input matrix B
    int* h_C = new int[N * N];   // host: CPU single-thread result
    int* h_C2 = new int[N * N];  // host: CPU multi-thread result
    int* h_C_gpu = new int[N * N];   // host: GPU naive result
    int* h_C_gpu2 = new int[N * N];  // host: GPU tiled result

    int *d_A, *d_B, *d_C;  // device pointers

    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * N * sizeof(int));
    cudaMalloc((void**)&d_C, N * N * sizeof(int));

    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() % 10;
        h_B[i] = rand() % 10;
        h_C[i] = 0;
        h_C2[i] = 0;
        h_C_gpu[i] = 0;
        h_C_gpu2[i] = 0;
    }

    // ======================
    // CPU TIMING (single thread)
    // ======================
    auto cpu_start = chrono::high_resolution_clock::now();

    cpu_mm_naive(h_A, h_B, h_C);

    auto cpu_end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> cpu_time = cpu_end - cpu_start;

    cout << "CPU time (single thread): " << cpu_time.count() << " ms\n";

    // ======================
    // CPU TIMING (multi thread)
    // ======================
    cpu_start = chrono::high_resolution_clock::now();

    unsigned int thread_count = cpu_mm_naive_mt(h_A, h_B, h_C2);

    cpu_end = chrono::high_resolution_clock::now();

    cpu_time = cpu_end - cpu_start;

    cout << "CPU time (multi thread: " << thread_count << " threads): " << cpu_time.count() << " ms\n";

    // ======================
    // GPU SETUP
    // ======================

    cudaMemcpy(d_A, h_A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 16;

    dim3 threadsperblock(block_size, block_size);  // 16x16 threads per block
    dim3 numBlocks((N + block_size - 1) / block_size,
                   (N + block_size - 1) / block_size);  // grid size to cover NxN

    // ======================
    // GPU TIMING (naive)
    // ======================

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    mm_naive<<<numBlocks, threadsperblock>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cout << "GPU kernel time(naive): " << gpu_time << " ms\n";

    cudaMemcpy(h_C_gpu, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // ======================
    // GPU TIMING (tiled)
    // ======================

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((int) N / TILE_SIZE, (int) N / TILE_SIZE);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMulTiled<<<blocks, threads>>>(d_A, d_B, d_C);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cout << "GPU kernel time(tiled): " << gpu_time << " ms\n";

    // ======================
    // Copy result back
    // ======================

    cudaMemcpy(h_C_gpu2, d_C, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    // ======================
    // Validate result
    // ======================

    long temp = 0;

    for (int i = 0; i < N * N; i++) {
        temp += abs(h_C_gpu[i] - h_C[i]);      // naive GPU vs CPU single
        temp += abs(h_C[i] - h_C2[i]);         // CPU single vs CPU multi
        temp += abs(h_C2[i] - h_C_gpu[i]);    // CPU multi vs naive GPU
        temp += abs(h_C_gpu[i] - h_C_gpu2[i]); // naive GPU vs tiled GPU
    }

    if (temp == 0)
        cout << "results are correct\n";
    else {
        cout << "wrong results\n";
        cout << temp << endl;
    }

    // ======================
    // Cleanup
    // ======================

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C2;
    delete[] h_C_gpu;
    delete[] h_C_gpu2;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

