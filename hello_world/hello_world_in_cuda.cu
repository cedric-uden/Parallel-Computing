// sourcecode copy/pasta from https://gpgpu.io/2019/12/07/cuda-hello-world/

#include <iostream>

// CUDA hello world kernel
__global__ void hello_world() {
    // printf is available for all GPUs with compute capability 2.0 and higher.
    printf("Hello World in CUDA!\n");
}

// program main fuction
int main(int argc, char *argv[]) {
    int exit = 0;

    // GPU hello world
    hello_world<<<1,1>>>();

    // This is not an explicit "flush buffer" function, but it serves that purpose here.
    // Without this call it is very likely there won't be output from the GPU.
    cudaDeviceSynchronize();
    // basic CUDA error checking
    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
        exit = 1;
    }
    return exit;
}