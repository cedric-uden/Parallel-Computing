#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <string.h>
#include "Runtime_Analysis.h"

#define RUN_GPU true
#define RUN_CPU true
#define VERIFY_ARRAYS_MATCH true


#define N 512        // Dimension: width and height of matrix

#define THREADS_PER_BLOCK_x 16
#define THREADS_PER_BLOCK_y 16

#define TARGET_INDEX (row * N + col)

__global__ void matrixMultiplyGPU(int *a, int *b, int *result) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
//        printf("%d,%d;", row, col);
        for (int k = 0; k < N; ++k) {
            result[TARGET_INDEX] = a[row * N + k] * b[k * N + col];
        }
    }
}

void matrixMultiplyCPU(int *a, int *b, int *result) {

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            for (int k = 0; k < N; ++k) {
                result[TARGET_INDEX] = a[row * N + k] * b[k * N + col];
            }
        }
    }
}

void initMatrix(int *a, int *b, int *result) {
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            a[TARGET_INDEX] = row;
            b[TARGET_INDEX] = col + 2;
            result[TARGET_INDEX] = 0;
        }
    }
}

int main() {
    int *a_cpu, *b_cpu, *result_cpu;
    int *a_gpu, *b_gpu, *result_gpu;
    int size = N * N * sizeof(int);

    if (RUN_GPU) {
        // Allocate memory on GPU
        cudaMallocManaged(&a_gpu, size);
        cudaMallocManaged(&b_gpu, size);
        cudaMallocManaged(&result_gpu, size);
        initMatrix(a_gpu, b_gpu, result_gpu);

        // prepare cuda values
        dim3 threads_per_block(THREADS_PER_BLOCK_x, THREADS_PER_BLOCK_y, 1);
        dim3 number_of_blocks((N / threads_per_block.x) + 1,
                              (N / threads_per_block.y) + 1,
                              1);    // Two-Dimensional Grid: z = 1

        // start with the timer
        auto *timer_gpu_total = new Runtime_Analysis("Timer GPU Total");
        timer_gpu_total->setStart();
        auto *timer_gpu = new Runtime_Analysis("Timer GPU Computation");
        timer_gpu->setStart();
        /////////////// compute matrix multiplication
        matrixMultiplyGPU <<<number_of_blocks, threads_per_block>>>(a_gpu, b_gpu, result_gpu);
        //
        timer_gpu->setEnd();
        std::cout << timer_gpu->print(TimerUnits::microseconds).rdbuf();

        auto *timer_sync = new Runtime_Analysis("Timer GPU Synchronize");
        timer_sync->setStart();
        /////////////// synchronize
        cudaDeviceSynchronize();
        //
        timer_sync->setEnd();
        std::cout << timer_sync->print(TimerUnits::microseconds).rdbuf();
        timer_gpu_total->setEnd();
        std::cout << timer_gpu_total->print(TimerUnits::microseconds).rdbuf();
    }


    if (RUN_CPU) {
        // Allocate memory on CPU
        a_cpu = (int *) malloc(size);
        b_cpu = (int *) malloc(size);
        result_cpu = (int *) malloc(size);
        initMatrix(a_cpu, b_cpu, result_cpu);

        // run matrix multiplication on cpu
        auto *timer_cpu = new Runtime_Analysis("TimerCPU");
        timer_cpu->setStart();
        matrixMultiplyCPU(a_cpu, b_cpu, result_cpu);
        timer_cpu->setEnd();
        std::cout << timer_cpu->print(TimerUnits::microseconds).rdbuf();
    }


    if (VERIFY_ARRAYS_MATCH) {
        //    printf("\n");
        bool error = false;

        auto *timer_comparison = new Runtime_Analysis("Timer compare both arrays");
        timer_comparison->setStart();
        for (int row = 0; row < N && !error; ++row) {
            for (int col = 0; col < N && !error; ++col) {
    //            printf("Values on \t\tGPU: %d\t\tCPU: %d\t\t\t", result_gpu[TARGET_INDEX], result_cpu[TARGET_INDEX]);
    //            printf("at index %d\n", TARGET_INDEX);
                char* error_message = "Error in matrix multiplication at position ";
                if (result_gpu[TARGET_INDEX] != result_cpu[TARGET_INDEX]) {
                    printf("%s[%d][%d]\n", error_message, row, col);
                    error = true;
                    break;
                }
            }
        }
        timer_comparison->setEnd();
        if (!error) {
            printf("Success! ");
        }
        std::cout << timer_comparison->print(TimerUnits::microseconds).rdbuf();
    }


    if (RUN_GPU) {
        cudaFree(a_gpu);
        cudaFree(b_gpu);
        cudaFree(result_gpu);
    }

    if (RUN_CPU) {
        free(a_cpu);
        free(b_cpu);
        free(result_cpu);
    }

    return 0;
}
