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


int getCoresPerSM(cudaDeviceProp &prop) {
    // Rough mapping by architecture
    switch (prop.major) {
        case 6:  // Pascal
            return 128;
        case 7:  // Volta/Turing
            return 64;
        case 8:  // Ampere
            return 128;
        case 9:  // Hopper
            return 128;
        default:
            std::cerr << "Unknown architecture, defaulting to 64 cores/SM\n";
            return 64;
    }
}

void printTheoreticalFLOPs(int device_id = 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);

    int sm_count = prop.multiProcessorCount;
    float clock_hz = prop.clockRate * 1000.0f;   // kHz → Hz
    int cores_per_sm = getCoresPerSM(prop);

    double flops_per_sec = (double)sm_count * cores_per_sm * 2 * clock_hz;
    double sec_per_op    = 1.0 / flops_per_sec;

    std::cout << "GPU name: " << prop.name << std::endl;
    std::cout << "Theoretical peak FP32: " 
              << flops_per_sec << " FLOPs/s" << std::endl;
    std::cout << "Seconds per FLOP: " 
              << sec_per_op << " sec/op" << std::endl;

    std::cout << std::endl;
}

int main() {

    //Add this to calcualte theoretical FLOPs
    printTheoreticalFLOPs();

    GPUTimer timer;


    int N = 8192;
    int d_k = 4096;

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

    for (int i = 0; i < 2; i++){
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
    }
    

    int T = 4;
    
    timer.tic();

    // Compute S = alpha * Q * K^T
    //   Q: (N x d_k)
    //   K: (N x d_k) -> K^T: (d_k x N)
    //   Result S: (N x N)
    for (int i = 0; i < T; i++){
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
    }
    ms = timer.toc() / T;
    std::cout << "(N, dk): " << N << ',' << d_k << std::endl;
    std::cout << "FLOP estimate: " << ((float)N*N*d_k*2) << std::endl;
    std::cout << "Matmul time: " << ms << " ms" << std::endl;
    std::cout << "(sec/FLOP): " << (ms*1e-03) / ((float)N*N*d_k*2) << std::endl;
    std::cout << std::endl;

    cublasDestroy(blas_handle);


    // Exponential
    int total = N * N;
    int numThreads = 256;
    int numBlocks = (total + numThreads - 1) / numThreads;
    
    timer.tic();
    for (int i = 0; i < T; i++){
        expKernel<<<numBlocks, numThreads>>>(d_S, total);
    }
    ms = timer.toc()/ T; 

    std::cout << "expKernel time: " << ms << " ms" << std::endl;
    std::cout << "(sec/FLOP): " << (ms*1e-03) / total << " " << std::endl;


    return 0;
}


// (N, dk): 8192,4096
// FLOP estimate: 5.49756e+11
// Matmul time: 57.374 ms
// (sec/FLOP): 1.04363e-13

// expKernel time: 1.53754 ms
// (sec/FLOP): 2.29111e-11 