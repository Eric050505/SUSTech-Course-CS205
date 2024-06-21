#include "matrix.h"
#include <immintrin.h>
#include <omp.h>
#include <cblas.h>
#include <cstdint>
#include <cstdio>
#include <stdint.h>
#include <cstring>
#define BLOCK_SIZE 64

Matrix create_matrix(size_t rows, size_t cols) {
    Matrix mat = {rows, cols, NULL};

    if (rows == 0 || cols == 0 || SIZE_MAX / rows < cols) {
        fprintf(stderr, "Invalid matrix dimensions %zu x %zu.\n", rows, cols);
        return mat;
    }

    mat.data = (float*) malloc(rows * cols * sizeof(float));
    if (mat.data == NULL) {
        fprintf(stderr, "Memory allocation failed for matrix size %zu x %zu.\n", rows, cols);
        return mat;
    }

    return mat;
}

void fill_matrix(Matrix* mat) {
    if (mat == NULL || mat->data == NULL) {
        fprintf(stderr, "Null pointer provided to fill_matrix.\n");
        return;
    }

    for (size_t i = 0; i < mat->rows * mat->cols; i++) {
        mat->data[i] = (float) rand() / RAND_MAX;
    }
}

void free_matrix(Matrix* mat) {
    if (mat == NULL) {
        fprintf(stderr, "Null pointer provided to free_matrix.\n");
        return;
    }
    free(mat->data);
    mat->data = NULL;
}

void matmul_plain(const Matrix* A, const Matrix* B, Matrix* C) {
    if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL) {
        fprintf(stderr, "Null pointer provided to matmul_plain.\n");
        return;
    }
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return;
    }

    for (size_t i = 0; i < A->rows; i++) {
        for (size_t j = 0; j < B->cols; j++) {
            float sum = 0.0;
            for (size_t k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C->data[i * C->cols + j] = sum;
        }
    }
}


void matmul_SIMD(const Matrix *A, const Matrix *B, Matrix *C) {
    if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL) {
        fprintf(stderr, "Null pointer provided to matmul_plain.\n");
        return;
    }
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return;
    }

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            __m256 sum = _mm256_setzero_ps();
            for (int k = 0; k < A->cols; k += 8) {
                __m256 a = _mm256_loadu_ps(&A->data[i * A->cols + k]);
                __m256 b = _mm256_loadu_ps(&B->data[k * B->cols + j]);
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }
            float buffer[8];
            _mm256_storeu_ps(buffer, sum);
            float total = buffer[0] + buffer[1] + buffer[2] + buffer[3] + buffer[4] + buffer[5] + buffer[6] + buffer[7];
            C->data[i * C->cols + j] = total;
        }
    }
}


void matmul_SIMD_16(const Matrix *A, const Matrix *B, Matrix *C) {
    if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL) {
        fprintf(stderr, "Null pointer provided to matmul_plain.\n");
        return;
    }
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return;
    }

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            __m512 sum = _mm512_setzero_ps();
            for (int k = 0; k < A->cols; k += 16) {
                __m512 a = _mm512_loadu_ps(&A->data[i * A->cols + k]);
                __m512 b = _mm512_loadu_ps(&B->data[k * B->cols + j]);
                sum = _mm512_add_ps(sum, _mm512_mul_ps(a, b));
            }
            float buffer[16];
            _mm512_storeu_ps(buffer, sum);
            float total = 0;
            for (int n = 0; n < 16; n++) {
                total += buffer[n];
            }
            C->data[i * C->cols + j] = total;
        }
    }
}


void matmul_SIMD_32(const Matrix *A, const Matrix *B, Matrix *C) {
    if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL) {
        fprintf(stderr, "Null pointer provided to matmul_plain.\n");
        return;
    }
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return;
    }

    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            for (int k = 0; k < A->cols; k += 32) {
                __m512 a0 = _mm512_loadu_ps(&A->data[i * A->cols + k]);
                __m512 a1 = _mm512_loadu_ps(&A->data[i * A->cols + k + 16]);
                __m512 b0 = _mm512_loadu_ps(&B->data[k * B->cols + j]);
                __m512 b1 = _mm512_loadu_ps(&B->data[(k + 16) * B->cols + j]);
                sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(a0, b0));
                sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(a1, b1));
            }
            sum0 = _mm512_add_ps(sum0, sum1);
            float buffer[16];
            _mm512_storeu_ps(buffer, sum0);
            float total = 0.0f;
            for (int n = 0; n < 16; n++) {
                total += buffer[n];
            }
            C->data[i * C->cols + j] = total;
        }
    }
}


void matmul_openMP(const Matrix *A, const Matrix *B, Matrix *C) {
    if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL) {
        fprintf(stderr, "Null pointer provided to matmul_plain.\n");
        return;
    }
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return;
    }

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            float sum = 0.0;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
            C->data[i * A->cols + j] = sum;
        }
    }
}


void matmul_SIMD_openMP(const Matrix *A, const Matrix *B, Matrix *C) {
    if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL) {
        fprintf(stderr, "Null pointer provided to matmul_plain.\n");
        return;
    }
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return;
    }

    #pragma omp parallel for
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            __m512 sum0 = _mm512_setzero_ps();
            __m512 sum1 = _mm512_setzero_ps();
            for (int k = 0; k < A->cols; k += 32) {
                __m512 a0 = _mm512_loadu_ps(&A->data[i * A->cols + k]);
                __m512 a1 = _mm512_loadu_ps(&A->data[i * A->cols + k + 16]);
                __m512 b0 = _mm512_loadu_ps(&B->data[k * B->cols + j]);
                __m512 b1 = _mm512_loadu_ps(&B->data[(k + 16) * B->cols + j]);
                sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(a0, b0));
                sum1 = _mm512_add_ps(sum1, _mm512_mul_ps(a1, b1));
            }
            sum0 = _mm512_add_ps(sum0, sum1);
            float buffer[16];
            _mm512_storeu_ps(buffer, sum0);
            float total = 0.0f;
            for (int n = 0; n < 16; n++) {
                total += buffer[n];
            }
            C->data[i * C->cols + j] = total;
        }
    }
}


void matmul_SIMD_openMP_block(const Matrix *A, const Matrix *B, Matrix *C) {
    if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL) {
        fprintf(stderr, "Null pointer provided to matmul_plain.\n");
        return;
    }
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return;
    }

    int i, j, k, ii, jj, kk;
    
    #pragma omp parallel for private(i, j, k, ii, jj, kk) collapse(2) schedule(static)
    for (i = 0; i < A->rows; i += BLOCK_SIZE) {
        for (j = 0; j < A->cols; j += BLOCK_SIZE) {
            for (k = 0; k < B->cols; k += BLOCK_SIZE) {
                for (ii = i; ii < i + BLOCK_SIZE && ii < A->rows; ii++) {
                    for (jj = j; jj < j + BLOCK_SIZE && jj < A->cols; jj++) {
                        __m512 sum0 = _mm512_setzero_ps();
                        for (kk = k; kk < k + BLOCK_SIZE && kk < B->cols; kk += 16) {
                            __m512 a0 = _mm512_loadu_ps(&A->data[ii * A->cols + kk]);
                            __m512 b0 = _mm512_loadu_ps(&B->data[kk * B->cols + jj]);
                            sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(a0, b0));
                        }
                        __m512 c0 = _mm512_loadu_ps(&C->data[ii * C->cols + jj]);
                        c0 = _mm512_add_ps(c0, sum0);
                        _mm512_storeu_ps(&C->data[ii * C->cols + jj], c0);
                    }
                }
            }
        }
    }
}



MortonMatrix::MortonMatrix(size_t n) : size(n), data(new float[n * n]()) {}
MortonMatrix::~MortonMatrix() {
    cleanup(this);
}
void cleanup(MortonMatrix* mat) {
    if (mat == NULL) {
        fprintf(stderr, "Null pointer provided to cleanup.\n");
        return;
    }
    delete[] mat->data;
    mat->data = NULL;
}
float& MortonMatrix::operator()(size_t row, size_t col) {
    return data[morton_index(row, col)];
}
size_t morton_index(size_t row, size_t col) {
    uint64_t z = 0;
    for (uint64_t i = 0; i < sizeof(uint32_t) * 8; i++) {
        z |= (row & (1ULL << i)) << i | (col & (1ULL << i)) << (i + 1);
    }
    return z;
}
void matmul_morton(const MortonMatrix& A, const MortonMatrix& B, MortonMatrix& C) {
    size_t size = A.size;
    const size_t blockSize = 256;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < size; i += blockSize) {
        for (size_t j = 0; j < size; j += blockSize) {
            for (size_t k = 0; k < size; k += blockSize) {
                for (size_t ii = i; ii < i + blockSize && ii < size; ++ii) {
                    for (size_t jj = j; jj < j + blockSize && jj < size; ++jj) {
                        __m512 sum0 = _mm512_setzero_ps();
                        for (size_t kk = k; kk < k + blockSize && kk < size; kk += 16) {
                            __m512 a0 = _mm512_loadu_ps(&A.data[morton_index(ii, kk)]);
                            __m512 b0 = _mm512_loadu_ps(&B.data[morton_index(kk, jj)]);
                            sum0 = _mm512_add_ps(sum0, _mm512_mul_ps(a0, b0));
                        }
                        __m512 c0 = _mm512_loadu_ps(&C.data[morton_index(ii, jj)]);
                        c0 = _mm512_add_ps(c0, sum0);
                        _mm512_storeu_ps(&C.data[morton_index(ii, jj)], c0);
                    }
                }
            }
        }
    }
}



void matmul_openBLAS(const Matrix *A, const Matrix *B, Matrix *C) {
    if (A == NULL || B == NULL || C == NULL || A->data == NULL || B->data == NULL || C->data == NULL) {
        fprintf(stderr, "Null pointer provided to matmul_plain.\n");
        return;
    }
    if (A->cols != B->rows || A->rows != C->rows || B->cols != C->cols) {
        fprintf(stderr, "Matrix dimensions do not match for multiplication.\n");
        return;
    }

    size_t M = A->rows, K = A->cols, N = B->cols;
    float alpha = 1.0, beta = 0.0;

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                M, N, K, 
                alpha, A->data, K, 
                B->data, N, 
                beta, C->data, N);
}




void add_matrix(Matrix* A, Matrix* B, Matrix* C) {
    for (size_t i = 0; i < A->rows; i += 16) {
        __m512 a = _mm512_loadu_ps(A + i);
        __m512 b = _mm512_loadu_ps(B + i);
        __m512 c = _mm512_add_ps(a, b);
        _mm512_storeu_ps(C + i, c);
    }
}
void subtract_matrix(Matrix* A, Matrix* B, Matrix* C) {
    for (size_t i = 0; i < A->rows; i += 16) {
        __m512 a = _mm512_loadu_ps(A + i);
        __m512 b = _mm512_loadu_ps(B + i);
        __m512 c = _mm512_sub_ps(a, b);
        _mm512_storeu_ps(C + i, c);
    }
}
void strassen_multiply(Matrix* A, Matrix* B, Matrix* C) {
    if (A->rows == 1 && A->cols == 1 && B->rows == 1 && B->cols == 1) {
        C->data[0] = A->data[0] * B->data[0];
        return;
    }

    size_t new_size = A->rows / 2;

    // Allocate memory for submatrices
    Matrix A11 = get_submatrix(A, 0, 0, new_size, new_size);
    Matrix A12 = get_submatrix(A, 0, new_size, new_size, 2 * new_size);
    Matrix A21 = get_submatrix(A, new_size, 0, 2 * new_size, new_size);
    Matrix A22 = get_submatrix(A, new_size, new_size, 2 * new_size, 2 * new_size);
    Matrix B11 = get_submatrix(B, 0, 0, new_size, new_size);
    Matrix B12 = get_submatrix(B, 0, new_size, new_size, 2 * new_size);
    Matrix B21 = get_submatrix(B, new_size, 0, 2 * new_size, new_size);
    Matrix B22 = get_submatrix(B, new_size, new_size, 2 * new_size, 2 * new_size);

    // Create matrices for storing intermediate results
    Matrix P1 = create_matrix(new_size, new_size);
    Matrix P2 = create_matrix(new_size, new_size);
    Matrix P3 = create_matrix(new_size, new_size);
    Matrix P4 = create_matrix(new_size, new_size);
    Matrix P5 = create_matrix(new_size, new_size);
    Matrix P6 = create_matrix(new_size, new_size);
    Matrix P7 = create_matrix(new_size, new_size);

    Matrix temp1 = create_matrix(new_size, new_size);
    Matrix temp2 = create_matrix(new_size, new_size);

    // P1 = A11 * (B12 - B22)
    subtract_matrix(&B12, &B22, &temp1);
    strassen_multiply(&A11, &temp1, &P1);

    // P2 = (A11 + A12) * B22
    add_matrix(&A11, &A12, &temp1);
    strassen_multiply(&temp1, &B22, &P2);

    // P3 = (A21 + A22) * B11
    add_matrix(&A21, &A22, &temp1);
    strassen_multiply(&temp1, &B11, &P3);

    // P4 = A22 * (B21 - B11)
    subtract_matrix(&B21, &B11, &temp1);
    strassen_multiply(&A22, &temp1, &P4);

    // P5 = (A11 + A22) * (B11 + B22)
    add_matrix(&A11, &A22, &temp1);
    add_matrix(&B11, &B22, &temp2);
    strassen_multiply(&temp1, &temp2, &P5);

    // P6 = (A12 - A22) * (B21 + B22)
    subtract_matrix(&A12, &A22, &temp1);
    add_matrix(&B21, &B22, &temp2);
    strassen_multiply(&temp1, &temp2, &P6);

    // P7 = (A11 - A21) * (B11 + B12)
    subtract_matrix(&A11, &A21, &temp1);
    add_matrix(&B11, &B12, &temp2);
    strassen_multiply(&temp1, &temp2, &P7);

    // Combine intermediate results into the result matrix C
    Matrix C11 = create_matrix(new_size, new_size);
    Matrix C12 = create_matrix(new_size, new_size);
    Matrix C21 = create_matrix(new_size, new_size);
    Matrix C22 = create_matrix(new_size, new_size);

    // C11 = P5 + P4 - P2 + P6
    add_matrix(&P5, &P4, &temp1);
    add_matrix(&temp1, &P6, &temp2);
    subtract_matrix(&temp2, &P2, &C11);

    // C12 = P1 + P2
    add_matrix(&P1, &P2, &C12);

    // C21 = P3 + P4
    add_matrix(&P3, &P4, &C21);

    // C22 = P5 + P1 - P3 - P7
    add_matrix(&P5, &P1, &temp1);
    subtract_matrix(&temp1, &P3, &temp2);
    subtract_matrix(&temp2, &P7, &C22);

    // Set submatrices back into the full matrix C
    set_submatrix(C, &C11, 0, 0);
    set_submatrix(C, &C12, 0, new_size);
    set_submatrix(C, &C21, new_size, 0);
    set_submatrix(C, &C22, new_size, new_size);

    // Free all allocated matrices
    free_matrix(&A11);
    free_matrix(&A12);
    free_matrix(&A21);
    free_matrix(&A22);
    free_matrix(&B11);
    free_matrix(&B12);
    free_matrix(&B21);
    free_matrix(&B22);
    free_matrix(&P1);
    free_matrix(&P2);
    free_matrix(&P3);
    free_matrix(&P4);
    free_matrix(&P5);
    free_matrix(&P6);
    free_matrix(&P7);
    free_matrix(&temp1);
    free_matrix(&temp2);
    free_matrix(&C11);
    free_matrix(&C12);
    free_matrix(&C21);
    free_matrix(&C22);
}
Matrix get_submatrix(Matrix* A, size_t x1, size_t y1, size_t x2, size_t y2) {
    Matrix sub;
    size_t rows = x2 - x1;
    size_t cols = y2 - y1;
    sub.rows = rows;
    sub.cols = cols;
    sub.data = (float*)malloc(rows * cols * sizeof(float));
    for (size_t i = 0; i < rows; ++i) {
        memcpy(sub.data + i * cols, A->data + (x1 + i) * A->cols + y1, cols * sizeof(float));
    }
    return sub;
}
void set_submatrix(Matrix* A, Matrix* src, size_t row, size_t col) {
    for (size_t i = 0; i < src->rows; ++i) {
        memcpy(A->data + ((row + i) * A->cols + col), src->data + i * src->cols, src->cols * sizeof(float));
    }
}