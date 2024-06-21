#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void createMatrix(float *m, int N) {
    for (int i = 0; i < N * N; i++)
        m[i] = (float)rand() / RAND_MAX;
}

void multiplyMatrices(const float *a, const float*b, float*c, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++) 
                c[i * N + j] += a[i * N + k] * b[k * N + j]; 
}

void optimized_I(const float*a, const float *b, float *c, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++)
                sum += a[i * N + k] * b[k * N + j];
            c[i* N + j] = sum;
        }
}

void optimized_II(const float *a, const float *b, float *c, int N) {
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++) {
            float r = a[i * N + k];
            for (int j = 0; j < N; j++) {
                c[i * N + j] += r * b[k * N + j];
            }
        }
}

void optimized_III(const float *a, const float *b, float *c, int N) {
    int i, j, k, ii, jj, kk;
    int BLOCK_SIZE = 50;
    if (N > 1500) BLOCK_SIZE = 100;
    #pragma omp parallel for private(i, j, k, ii, jj, kk) shared(a, b, c, N)
    for (ii = 0; ii < N; ii += BLOCK_SIZE)
        for (jj = 0; jj < N; jj += BLOCK_SIZE)
            for (kk = 0; kk < N; kk += BLOCK_SIZE)
                for (i = ii; i < ii + BLOCK_SIZE && i < N; i++)
                    for (k = kk; k < kk + BLOCK_SIZE && k < N; k++) {
                        float r = a[i * N + k];
                        for (j = jj; j < jj + BLOCK_SIZE && j < N; j++)
                            c[i * N + j] += r * b[k * N + j];
                    }
}

int main() {
    for (int i = 0; i <= 2000; i += 100) {
        float *a = (float*)malloc(i * i * sizeof(float)), *b = (float*)malloc(i * i * sizeof(float)), *c = (float*)malloc(i * i * sizeof(float));
        createMatrix(a, i);
        createMatrix(b, i);
        createMatrix(a, i);
        clock_t startTime = clock();
        multiplyMatrices(a, b, c, i);
        clock_t endTime = clock();
        double timeCost = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
        printf("The Size of Matrices: %d\nTime Taken: %f ms", i, timeCost);
        startTime = clock();
        optimized_I(a, b, c, i);
        endTime = clock();
        timeCost = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
        printf("\nTime Taken after the first Optimization: %f ms", timeCost);
        startTime = clock();
        optimized_II(a, b, c, i);
        endTime = clock();
        timeCost = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
        printf("\nTime Taken after the second Optimization: %f ms", timeCost);
        startTime = clock();
        optimized_III(a, b, c, i);
        endTime = clock();
        timeCost = (double)(endTime - startTime) / CLOCKS_PER_SEC * 1000;
        printf("\nTime Taken after the third Optimization: %f ms", timeCost);
        printf("\n-------------------------------");
        free(a);
        free(b);
        free(c);
        printf("\n");
    }
}