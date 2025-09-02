#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float* A, const float* B, float* C, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N){
        C[idx] = A[idx] + B[idx];
    }

}
  

int main(){
    const int N = 1024;
    const int size = N * sizeof(int);

    float *v_A = new float[N];
    float *v_B = new float[N];
    float *v_C = new float[N];

    for (int i = 0; i < N; i++){
        v_A[i] = 1;
        v_B[i] = i + 1.5;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size); // GPU space for A
    cudaMalloc((void**)&d_B, size); // GPU space for B
    cudaMalloc((void**)&d_C, size); // GPU space for C

    // copy input data from host to device:
    cudaMemcpy(d_A, v_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, v_B, size, cudaMemcpyHostToDevice);

    int numThreads = 256;
    int numBlocks = (N + numThreads - 1) / numThreads;

    vectorAdd<<<numBlocks, numThreads>>>(d_A, d_B, d_C, N);

    cudaMemcpy(v_C, d_C, size, cudaMemcpyDeviceToHost);

    for(int i = N-10;i<N;i++){
        std::cout << "C[" << i << "] = " << v_C[i] << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    delete[] v_A;
    delete[] v_B;
    delete[] v_C;

    return 0;

}