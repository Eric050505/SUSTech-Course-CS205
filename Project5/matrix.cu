#include "matrix.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

void initMat(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

__global__ void matTransKernelAdv(const float* __restrict__ A, float a, float b, float* B, int rows, int cols) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;

    int threadIdxInBlock = threadIdx.x + threadIdx.y * blockDim.x;

    extern __shared__ float shared_memory[];
    float* shared_A = shared_memory;

    if (idx < cols && idy < rows) {
        shared_A[threadIdxInBlock] = A[idy * cols + idx];
    }
    
    __syncthreads();



    if (idx < cols && idy < rows) {
        B[idy * cols + idx] = a * shared_A[threadIdxInBlock] + b;
    }
}

void matTrans(const float* A, float a, float b, float* B, int rows, int cols, dim3 threadsPerBlock) {
    float *data_A, *data_B;
    int size = rows * cols * sizeof(float);

    cudaMalloc((void**)&data_A, size);
    cudaMalloc((void**)&data_B, size);

    cudaMemcpy(data_A, A, size, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matTransKernelAdv<<<blocksPerGrid, threadsPerBlock>>>(data_A, a, b, data_B, rows, cols);
    cudaMemcpy(B, data_B, size, cudaMemcpyDeviceToHost);

    cudaFree(data_A);
    cudaFree(data_B);
}

void matTransAdv(const float* A, float a, float b, float* B, int rows, int cols, dim3 threadsPerBlock) {
    float *data_A, *data_B;
    int size = rows * cols * sizeof(float);

    cudaMalloc((void**)&data_A, size);
    cudaMalloc((void**)&data_B, size);

    cudaMemcpy(data_A, A, size, cudaMemcpyHostToDevice);

    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    int sharedMemSize = threadsPerBlock.x * threadsPerBlock.y * sizeof(float);

    matTransKernelAdv<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(data_A, a, b, data_B, rows, cols);
    
    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in kernel launch: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaMemcpy(B, data_B, size, cudaMemcpyDeviceToHost);

    cudaFree(data_A);
    cudaFree(data_B);
}

void cublasSgeamTrans(const float* A, float a, float b, float* B, int rows, int cols) {
    float *d_A, *d_B;
    int size = rows * cols * sizeof(float);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemset(d_B, 0, size);
    float alpha = a;
    float beta = b;

    cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, &alpha, d_A, rows, &beta, d_B, rows, d_B, rows);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cublasDestroy(handle);
}

void cublasSgemmTrans(const float* A, float a, float b, float* B, int rows, int cols) {
    float *d_A, *d_B;
    int size = rows * cols * sizeof(float);

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemset(d_B, 0, size);

    float alpha = a;
    float beta = b;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, rows, cols, 1, &alpha, d_A, rows, d_B, rows, &beta, d_B, rows);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cublasDestroy(handle);
}

void matTransCPU(const float* A, float a, float b, float* B, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            B[index] = a * A[index] + b;
        }
    }
}
