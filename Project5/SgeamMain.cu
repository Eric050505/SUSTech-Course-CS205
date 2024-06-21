#include <chrono>
#include "matrix.h"




int main() {
    // Define scalar values
    float a = 2.0f;
    float b = 1.0f;
    
    int sizes[] = {512, 1024, 2048, 4096, 8192, 16384, 32768, 32768*2};

    for (int matSize : sizes) {
        auto start = std::chrono::high_resolution_clock::now();
        float *A, *B;
        int size = matSize * matSize;
        A = (float*)malloc(size * sizeof(float));
        B = (float*)malloc(size * sizeof(float));
        initMat(A, matSize, matSize);


        cublasSgeamTrans(A, a, b, B, matSize, matSize);
        

        free(A);
        free(B);
        auto stop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = stop - start;
        std::cout << "cublasSgeamTrans - Matrix Scalar Add of size " << matSize << "x" << matSize << " took " << duration.count() << " ms." << std::endl;    
    }


    return 0;
}