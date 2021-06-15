#include <stdio.h>
#include <assert.h>

#define TARGET_INDEX (zeile * N + spalte)
#define N 64        // Dimension, Hoehe und Breite der Matritzen

__global__ void matrixMultiplyGPU(int *a, int *b, int *ergebnis) {
    int akkumulator = 0;

    int zeile = blockIdx.x * blockDim.x + threadIdx.x;
    int spalte = blockIdx.y * blockDim.y + threadIdx.y;

    if (zeile < N && spalte < N) {
        printf("%d,%d;", zeile, spalte);
        for (int k = 0; k < N; ++k) {
            akkumulator += a[zeile * N + k] * b[k * N + spalte];
        }
        ergebnis[TARGET_INDEX] = akkumulator;
    }
}

void matrixMultiplyCPU(int *a, int *b, int *ergebnis) {
    int akkumulator = 0;    // Ergebnis-"Akkumulator" für Zwischenwerte

    for (int zeile = 0; zeile < N; ++zeile) {
        for (int spalte = 0; spalte < N; ++spalte) {
            for (int k = 0; k < N; ++k) {
                akkumulator += a[zeile * N + k] * b[k * N + spalte];
            }
            ergebnis[TARGET_INDEX] = akkumulator;
        }
    }
}

int main() {
    int *a_cpu, *b_cpu, *ergebnis_cpu;
    int *a_gpu, *b_gpu, *ergebnis_gpu;
    int size = N * N * sizeof(int);

    // Allocate memory on CPU
    a_cpu = (int *) malloc(size);
    b_cpu = (int *) malloc(size);
    ergebnis_cpu = (int *) malloc(size);

    // Allocate memory on GPU
    cudaMallocManaged(&a_gpu, size);
    cudaMallocManaged(&b_gpu, size);
    cudaMallocManaged(&ergebnis_gpu, size);

    // Initialisieren der Matritzen
    for (int zeile = 0; zeile < N; ++zeile) {
        for (int spalte = 0; spalte < N; ++spalte) {
            a_cpu[TARGET_INDEX] = zeile;
            b_cpu[TARGET_INDEX] = spalte + 2;
            ergebnis_cpu[TARGET_INDEX] = 0;
            ergebnis_gpu[TARGET_INDEX] = 0;
        }
    }

    dim3 threads_per_block(26, 16, 1);    // 16 x 16 Block-Threads
    dim3 number_of_blocks((N / threads_per_block.x) + 1,
                          (N / threads_per_block.y) + 1,
                          1);    //Zweidimensionales Grid, z = 1

    matrixMultiplyGPU <<<number_of_blocks, threads_per_block>>>(a_gpu, b_gpu,
                                                                ergebnis_gpu);

    cudaDeviceSynchronize();

    printf("\n");

    matrixMultiplyCPU(a_cpu, b_cpu, ergebnis_cpu);

    bool error = false;

    for (int zeile = 0; zeile < N && !error; ++zeile) {

        for (int spalte = 0; spalte < N && !error; ++spalte) {

            printf("Values on \t\tGPU: %d\t\tCPU: %d\t\t\t", ergebnis_gpu[TARGET_INDEX], ergebnis_cpu[TARGET_INDEX]);
            printf("at index %d\n", TARGET_INDEX);

            if (ergebnis_gpu[TARGET_INDEX] !=
                ergebnis_cpu[TARGET_INDEX]) {
                printf("Fehler in Matrixmultiplikation an der Stelle ergebnis[%d][%d]\n",
                       zeile, spalte);
                error = true;
                break;
            }
        }
        if (!error) {
            printf("Erfolg!\n");
        }
    }

    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(ergebnis_gpu);

    free(a_cpu);
    free(b_cpu);
    free(ergebnis_cpu);

    return 0;
}
