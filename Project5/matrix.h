#ifndef MATRIX_H
#define MATRIX_H

#include <cuda_runtime.h>
#include <iostream>

void initMat(float* matrix, int rows, int cols);
void matTrans(const float* A, float a, float b, float* B, int rows, int cols, dim3 threadsPerBlock);
void cublasSgeamTrans(const float* A, float a, float b, float* B, int rows, int cols);
void cublasSgemmTrans(const float* A, float a, float b, float* B, int rows, int cols);
void matTransAdv(const float* A, float a, float b, float* B, int rows, int cols, dim3 threadsPerBlock);
void matTransCPU(const float* A, float a, float b, float* B, int rows, int cols);

#endif // MATRIX_H
