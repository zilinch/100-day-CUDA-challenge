#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <chrono>

#include <iostream>
#include <vector>
#include <cmath>


struct GPUTimer {
    cudaEvent_t start, stop;
    GPUTimer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    ~GPUTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    void tic() {
        cudaEventRecord(start);
    }
    float toc() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        return ms;
    }
};

__global__ void expKernel(float *S, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    
    if(idx < N){
        S[idx] = expf(S[idx]);
    }
}

int main() {

    GPUTimer timer;

    int N = 2048;
    int d_k = 128;

    size_t size_QK = (size_t)N * d_k * sizeof(float);
    size_t size_S = (size_t)N * N * sizeof(float);
    
    std::vector<float> h_Q(N * d_k), h_K(N* d_k);

    for (int i = 0; i < N * d_k; i++){
        h_Q[i] = (float)rand() / RAND_MAX;
        h_K[i] = (float)rand() / RAND_MAX;
    }

    float *d_Q, *d_K, *d_S;
    cudaMalloc((void**)&d_Q, size_QK);
    cudaMalloc((void**)&d_K, size_QK);
    cudaMalloc((void**)&d_S, size_S);

    // use CudaEvent Timing
    timer.tic();
    cudaMemcpy(d_Q, h_Q.data(), size_QK, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K.data(), size_QK, cudaMemcpyHostToDevice);
    float ms = timer.toc();
    // std::cout << "Memcpy time: " << ms << " ms" << std::endl;

    // // use CPU Wall Clock Timing
    // auto cpu_start = std::chrono::high_resolution_clock::now();
    // cudaMemcpy(d_Q, h_Q.data(), size_QK, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_K, h_K.data(), size_QK, cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize(); 
    // auto cpu_end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> cpu_ms = cpu_end - cpu_start;
    // std::cout << "CPU chrono memcpy time: " << cpu_ms.count() << " ms" << std::endl;

    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);

    float alpha = 1.0f / std::sqrt((float) d_k);
    float beta  = 0.0f;

    cublasSgemm(
        blas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, d_k, 
        &alpha,
        d_K, d_k,
        d_Q, d_k,   
        &beta,
        d_S, N      
    );


    timer.tic();

    // Compute S = alpha * Q * K^T
    //   Q: (N x d_k)
    //   K: (N x d_k) -> K^T: (d_k x N)
    //   Result S: (N x N)
    cublasSgemm(
        blas_handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, N, d_k, 
        &alpha,
        d_K, d_k,
        d_Q, d_k,   
        &beta,
        d_S, N      
    );

    // cudaDeviceSynchronize(); // wait until GEMM finishes
    ms = timer.toc();
    std::cout << "GEMM time: " << ms << " ms" << std::endl;
    // std::cout << "Debug: " << (float)N*N*d_k << " " << std::endl;
    std::cout << "Rate: " << (ms*1e-03) / ((float)N*N*d_k) << " " << std::endl;

    cublasDestroy(blas_handle);

    int total = N * N;
    int numThreads = 256;
    int numBlocks = (total + numThreads - 1) / numThreads;
    
    timer.tic();
    expKernel<<<numBlocks, numThreads>>>(d_S, total);
    // cudaDeviceSynchronize();
    ms = timer.toc(); //e^S

    std::cout << "expKernel time: " << ms << " ms" << std::endl;
    std::cout << "Rate: " << (ms*1e-03) / total << " " << std::endl;


    return 0;
}