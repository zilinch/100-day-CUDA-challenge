#include <cuda_runtime.h>
#include <iostream>
#include <vector>



__global__ void fma_kernel(float *out, int n) {
    float a = 1.0f, b = 2.0f, c = 3.0f, d = 0.0f;
    for (int i = 0; i < n; i++) {
        d = fmaf(a, b, c); // FMA = 2 FLOPs, guaranteed fused
    }
    out[threadIdx.x + blockIdx.x * blockDim.x] = d;
}


__global__ void fma_kernel_4(float *out, int n) {
    float a = 1.0f, b = 2.0f;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f;

    for (int i = 0; i < n; i++) {
        c0 = fmaf(a, b, c0);  
        c1 = fmaf(a, b, c1);  
        c2 = fmaf(a, b, c2);  
        c3 = fmaf(a, b, c3);  
    }

    float d = c0 + c1 + c2 + c3;
    out[threadIdx.x + blockIdx.x * blockDim.x] = d;
}


__global__ void fma_kernel_7(float *out, int n) {
    float a = 1.0f, b = 2.0f;
    float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;
    float c3 = 0.0f, c4 = 0.0f, c5 = 0.0f, c6 = 0.0f;

    for (int i = 0; i < n; i++) {
        c0 = fmaf(a, b, c0);
        c1 = fmaf(a, b, c1);
        c2 = fmaf(a, b, c2);
        c3 = fmaf(a, b, c3);
        c4 = fmaf(a, b, c4);
        c5 = fmaf(a, b, c5);
        c6 = fmaf(a, b, c6);
    }

    float d = c0 + c1 + c2 + c3 + c4 + c5 + c6;
    out[threadIdx.x + blockIdx.x * blockDim.x] = d;
}


__global__ void fma_kernel_8(float *out, int n) {
    float a = 1.0f, b = 2.0f;
    float c0=0,c1=0,c2=0,c3=0,c4=0,c5=0,c6=0,c7=0;
    for (int i = 0; i < n; i++) {
        c0 = fmaf(a, b, c0);
        c1 = fmaf(a, b, c1);
        c2 = fmaf(a, b, c2);
        c3 = fmaf(a, b, c3);
        c4 = fmaf(a, b, c4);
        c5 = fmaf(a, b, c5);
        c6 = fmaf(a, b, c6);
        c7 = fmaf(a, b, c7);
    }
    float d = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7;
    out[threadIdx.x + blockIdx.x * blockDim.x] = d;
}


// Host code
int main() {

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device Name: " << prop.name << std::endl;
    std::cout << "SM count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per SM: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;



    int sm_count = prop.multiProcessorCount;
    int threads_per_block = 256; //Four blocks per SM
    int numBlocks = (sm_count * 32); // 
    int numTreads = numBlocks * threads_per_block;

    float *d_out;
    cudaMalloc((void**)&d_out, numTreads * sizeof(float));


    int n_iter = 8192;    // loop count inside kernel
    int accumulators = 8;


    // Launch kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    // fma_kernel<<<numBlocks, threads_per_block>>>(d_out, n_iter);
    // fma_kernel_4<<<numBlocks, threads_per_block>>>(d_out, n_iter);
    fma_kernel_8<<<numBlocks, threads_per_block>>>(d_out, n_iter);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    long long flops_per_thread = (long long)n_iter * 2 * accumulators; // FMA = 2 FLOPs
    long long total_flops = numTreads * flops_per_thread;

    double tflops = (double)total_flops / (ms / 1000.0) / 1.0e12;

    std::cout << "Achieved: " << tflops << " TFLOPs/s" << std::endl;

    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    return 0;
}