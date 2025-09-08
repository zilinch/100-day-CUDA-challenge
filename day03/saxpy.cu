#include <cuda_runtime.h>
#include <cublas_v2.h>

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

__global__ void saxpyKernel(const float *A, const float *B, float *C, float alpha, int maxLength){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < maxLength){
        C[idx] = alpha * A[idx] + B[idx];
    }
}


int main() {

    GPUTimer timer;

    int size_C = 4096;
    int size_R = 4096;
    int size_Mat = size_C * size_R;
    size_t bytes = size_Mat * sizeof(float);

    float alpha = 2.0f;

    std::vector<float> h_A(size_Mat), h_B(size_Mat), h_C(size_Mat);

    for (int i = 0; i < size_Mat; i ++) {
        h_A[i] = 1.0f;
        h_B[i] = static_cast<float>(i);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);


    int numThreads = 256;
    int numBlocks = (size_Mat + numThreads - 1) / numThreads;

    timer.tic();
    saxpyKernel<<<numBlocks, numThreads>>>(d_A, d_B, d_C, alpha, size_Mat);
    float gpuTime = timer.toc();
    std::cout << "MyKernel GPU time: " << gpuTime << " ms" << std::endl;

    // cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    // for (int i = 0; i < 10; i++) {
    //     std::cout << "C[" << i << "] = " << h_C[i] << std::endl;
    // }

    // --- cuBLAS SAXPY timing ---
    cublasHandle_t handle;
    cublasCreate(&handle);
    cudaMemcpy(d_C, d_B, bytes, cudaMemcpyDeviceToDevice); // C = B

    int T = 4;
    timer.tic();
    for (int t = 0; t < T; t++){
        cublasSaxpy(handle, size_Mat, &alpha, d_A, 1, d_C, 1);
    }
    gpuTime = timer.toc()/T;

    std::cout << "cuBLAS GPU time: " << gpuTime << " ms" << std::endl;
    // cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(handle);

    return 0;
}