#include <cstdio>
__global__ void add(int a,int b,int* c){ *c = a + b; }
int main(){
  int *d_c, h_c;
  cudaMalloc(&d_c, sizeof(int));
  add<<<1,1>>>(2,3,d_c);
  cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);
  printf("2 + 3 = %d\n", h_c);
  cudaFree(d_c);
  return 0;
}