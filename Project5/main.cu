#include <iostream>
#include <cuda_runtime.h>
#include "matrix.h"
#include <chrono>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    float a = 2.0f;
    float b = 1.0f;
    
    int sizes[] = {512, 1024, 2048, 4096, 8192, 16384};

    for (int matSize : sizes) {
        
        std::cout << std::endl;
        float *A, *B;
        int size = matSize * matSize;
        A = (float*)malloc(size * sizeof(float));
        B = (float*)malloc(size * sizeof(float));
        initMat(A, matSize, matSize);

       // cudaEvent_t start, stop;
       // CUDA_CHECK(cudaEventCreate(&start));
       // CUDA_CHECK(cudaEventCreate(&stop));
       // CUDA_CHECK(cudaEventRecord(start, 0));
        auto start = std::chrono::high_resolution_clock::now();

        // matTrans(A, a, b, B, matSize, matSize, dim3(32, 32));
        matTransAdv(A, a, b, B, matSize, matSize);

        auto stop = std::chrono::high_resolution_clock::now();
        //  CUDA_CHECK(cudaEventRecord(stop, 0));
        //  CUDA_CHECK(cudaEventSynchronize(stop));

        //  float elapsedTime;
        //      CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
       
        
        //    CUDA_CHECK(cudaEventDestroy(start));
        //    CUDA_CHECK(cudaEventDestroy(stop));

        free(A);
        free(B);
        
        std::chrono::duration<float, std::milli> duration = stop - start;
        std::cout << "matTrans - Matrix Scalar Add of size " << matSize << "x" << matSize << " took " << duration.count() << " ms." << std::endl;    
    }
    return 0;
}


