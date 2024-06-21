#ifndef MATRIX_H
#define MATRIX_H

#include <stdlib.h>
#include <stdint.h>

typedef struct Matrix {
    size_t rows;
    size_t cols;
    float* data;
} Matrix;

struct MortonMatrix {
    size_t size;
    float* data;
    MortonMatrix(size_t n);
    ~MortonMatrix();
    float& operator()(size_t row, size_t col);
};





Matrix create_matrix(size_t rows, size_t cols);
void fill_matrix(Matrix* mat);
void free_matrix(Matrix* mat);
void matmul_plain(const Matrix* A, const Matrix* B, Matrix* C);
void matmul_SIMD(const Matrix* A, const Matrix* B, Matrix* C);
void matmul_SIMD_16(const Matrix* A, const Matrix* B, Matrix* C);
void matmul_SIMD_32(const Matrix* A, const Matrix* B, Matrix* C);
void matmul_openMP(const Matrix* A, const Matrix* B, Matrix* C);
void matmul_SIMD_openMP(const Matrix* A, const Matrix* B, Matrix* C);
void matmul_SIMD_openMP_block(const Matrix* A, const Matrix* B, Matrix* C);
void matmul_morton(const MortonMatrix& A, const MortonMatrix& B, MortonMatrix& C);
size_t morton_index(size_t row, size_t col);
void cleanup(MortonMatrix* mat);



void matmul_openBLAS(const Matrix *A, const Matrix *B, Matrix *C);



void add_matrix(Matrix* A, Matrix* B, Matrix* result);
void subtract_matrix(Matrix* A, Matrix* B, Matrix* result);
void strassen_multiply(Matrix* A, Matrix* B, Matrix* C);
Matrix get_submatrix(Matrix* A, size_t x1, size_t y1, size_t x2, size_t y2);
void set_submatrix(Matrix* A, Matrix* src, size_t row, size_t col);


#endif