
//reference: https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-matrix-multiplication-no-shared-memory

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>

#define BLOCK_SIZE 16

typedef struct {
    int width;
    int height;
    int stride;
    float *elements;
} Matrix;


__device__ float GetElement(const Matrix A, int row, int col) {
    return A.elements[row * A.stride + col];
}

__device__ void SetElement(Matrix A, int row, int col, float value) {
    A.elements[row * A.stride + col] = value;
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) {
    Matrix Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * row * BLOCK_SIZE + BLOCK_SIZE * col];
    return Asub;
}

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);


//host code
void Matmul(const Matrix A, const Matrix B, Matrix C){

    Matrix d_A, d_B, d_C;
    d_A.width = d_A.stride = A.width; d_A.height = A.height;
    size_t size_A = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size_A);
    cudaMemcpy(d_A.elements, A.elements, size_A, cudaMemcpyHostToDevice);

    d_B.width = d_B.stride = B.width; d_B.height = B.height;
    size_t size_B = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size_B);
    cudaMemcpy(d_B.elements, B.elements, size_B, cudaMemcpyHostToDevice);

    d_C.width = d_C.stride = C.width; d_C.height = C.height;
    size_t size_C = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size_C);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); // 16x16 threads per block
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y); // number of blocks

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); //warm up

    const int runs = 5;
    float total_ms = 0.0f;
    for (int i = 0; i < runs; i++) {
        cudaEventRecord(start);

        MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms += ms;
    }

    float avg_ms = total_ms / runs;
    double flops = 2.0 * (double)size_C * (double)A.width;
    double tflops = flops / (avg_ms / 1000.0) / 1.0e12;

    std::cout << "Avg time: " << avg_ms << " ms" << std::endl;
    std::cout << "Achieved: " << tflops << " TFLOPs/s" << std::endl;


    cudaMemcpy(C.elements, d_C.elements, size_C, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}


__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    float Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {
        Matrix Asub = GetSubMatrix(A, blockRow, m);
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        __syncthreads();

        for (int e = 0; e < BLOCK_SIZE; ++e) {
            Cvalue += As[row][e] * Bs[e][col];
        }

        __syncthreads();
    }
    SetElement(Csub, row, col, Cvalue);  
}


int main() {
    
    const int N = 256, M = 256, K = 256;

    Matrix A, B, C;
    A.width = K; A.height = N; A.stride = K;
    B.width = M; B.height = K; B.stride = M;
    C.width = M; C.height = N; C.stride = M;

    size_t size_A = A.width * A.height;
    size_t size_B = B.width * B.height;
    size_t size_C = C.width * C.height;

    A.elements = (float *)malloc(size_A * sizeof(float));
    B.elements = (float *)malloc(size_B * sizeof(float));
    C.elements = (float *)malloc(size_C * sizeof(float));

    
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);

    for (int i = 0; i < size_A; ++i) {
        A.elements[i] = dis(gen);
    }

    for (int i = 0; i < size_B; ++i) {
        B.elements[i] = dis(gen);
    }


    Matmul(A, B, C);

    // std::cout << "First 16x16 block of C:" << std::endl;
    // for (int i = 0; i < 16; i++) {                 // first 16 rows
    //     for (int j = 0; j < 16; j++) {             // first 16 cols
    //         std::cout << C.elements[i * C.width + j] << "\t";
    //     }
    //     std::cout << std::endl;
    // }

    free(A.elements);
    free(B.elements);
    free(C.elements);
    return 0;
}