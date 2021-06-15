#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string.h>
#include "Runtime_Analysis.h"

#define TARGET_INDEX (row * N + column)
#define N 64        // Dimension: width and height of matrix

__global__ void matrixMultiplyGPU(int *a, int *b, int *result) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int column = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && column < N) {
//        printf("%d,%d;", row, column);
        for (int k = 0; k < N; ++k) {
            result[TARGET_INDEX] = a[row * N + k] * b[k * N + column];
        }
    }
}

void matrixMultiplyCPU(int *a, int *b, int *result) {

    for (int row = 0; row < N; ++row) {
        for (int column = 0; column < N; ++column) {
            for (int k = 0; k < N; ++k) {
                result[TARGET_INDEX] = a[row * N + k] * b[k * N + column];
            }
        }
    }
}

int main() {
    int *a_cpu, *b_cpu, *result_cpu;
    int *a_gpu, *b_gpu, *result_gpu;
    int size = N * N * sizeof(int);

    // Allocate memory on CPU
    a_cpu = (int *) malloc(size);
    b_cpu = (int *) malloc(size);
    result_cpu = (int *) malloc(size);

    // Allocate memory on GPU
    cudaMallocManaged(&a_gpu, size);
    cudaMallocManaged(&b_gpu, size);
    cudaMallocManaged(&result_gpu, size);

    // Initialize Matrices
    for (int row = 0; row < N; ++row) {
        for (int column = 0; column < N; ++column) {
            a_cpu[TARGET_INDEX] = row;
            b_cpu[TARGET_INDEX] = column + 2;
            result_cpu[TARGET_INDEX] = 0;
            a_gpu[TARGET_INDEX] = row;
            b_gpu[TARGET_INDEX] = column + 2;
            result_gpu[TARGET_INDEX] = 0;
        }
    }


    dim3 threads_per_block(16, 16, 1);    // 16 x 16 Block-Threads
    dim3 number_of_blocks((N / threads_per_block.x) + 1,
                          (N / threads_per_block.y) + 1,
                          1);    // Two-Dimensional Grid: z = 1

    auto *timer_gpu = new Runtime_Analysis("TimerGPU");
    timer_gpu->setStart();
    matrixMultiplyGPU <<<number_of_blocks, threads_per_block>>>(a_gpu, b_gpu,
                                                                result_gpu);
    timer_gpu->setEnd();
    std::cout << timer_gpu->print(TimerUnits::microseconds).rdbuf();

    cudaDeviceSynchronize();

//    printf("\n");

    auto *timer_cpu = new Runtime_Analysis("TimerCPU");
    timer_cpu->setStart();
    matrixMultiplyCPU(a_cpu, b_cpu, result_cpu);
    timer_cpu->setEnd();
    std::cout << timer_cpu->print(TimerUnits::microseconds).rdbuf();

    bool error = false;


    auto *timer_comparison = new Runtime_Analysis("Timer compare both arrays");
    timer_comparison->setStart();
    for (int row = 0; row < N && !error; ++row) {

        for (int column = 0; column < N && !error; ++column) {

//            printf("Values on \t\tGPU: %d\t\tCPU: %d\t\t\t", result_gpu[TARGET_INDEX], result_cpu[TARGET_INDEX]);
//            printf("at index %d\n", TARGET_INDEX);

            if (result_gpu[TARGET_INDEX] !=
                result_cpu[TARGET_INDEX]) {
                printf("Error in matrix multiplication at position[%d][%d]\n",
                       row, column);
                error = true;
                break;
            }
        }
    }
    timer_comparison->setEnd();
    std::cout << timer_comparison->print(TimerUnits::microseconds).rdbuf();
    if (!error) {
        printf("Success!\n");
    }

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(result_gpu);

    free(a_cpu);
    free(b_cpu);
    free(result_cpu);

    return 0;
}
