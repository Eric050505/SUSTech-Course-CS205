#include <iostream>
#include <chrono>
#include <thread>  
#include "matrix.h"


int main() {
    size_t sizes[] = {2048, 4196, 8192, 16384 ,65536};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; ++i) {
        printf("Size: %d\n", (int)sizes[i]);
        size_t size = sizes[i];
        
        Matrix A = create_matrix(size, size);
        Matrix B = create_matrix(size, size);
        Matrix C = create_matrix(size, size);
        
        
        fill_matrix(&A);
        fill_matrix(&B);

        auto start = std::chrono::high_resolution_clock::now();
        matmul_SIMD_openMP_block(&A, &B, &C);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Time taken by function: " << (float)duration.count() / 1000 << " seconds" << std::endl;

        free_matrix(&A);
        free_matrix(&B);
        free_matrix(&C);

        /*      
        MortonMatrix A(matrixSize), B(matrixSize), C(matrixSize);
        for (size_t i = 0; i < matrixSize; ++i) {
            for (size_t j = 0; j < matrixSize; ++j) {
                A(i, j) = static_cast<float>(i + j);
                B(i, j) = static_cast<float>(i - j);
            }
        }
  
        start = clock();
        matmul_SIMD_16(&A, &B, &C);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD_16 Time Cost: %f\n", cpu_time_used);

        start = clock();
        matmul_SIMD_32(&A, &B, &C);
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("SIMD_32 Time Cost: %f\n", cpu_time_used);
        cleanup(&A);
        cleanup(&B);
        cleanup(&C);*/




    }

    return 0;
}
