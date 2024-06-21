#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <memory>
#include <cstring>
#include <stdexcept>

template<typename T>
class Matrix;

template<typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix);

template<typename T>
std::istream &operator>>(std::istream &is, Matrix<T> &matrix);

template<typename T>
class Matrix {
private:

    size_t rows;
    size_t cols;
    std::shared_ptr<T[]> data;
    bool isROI;
    Matrix<T> *parent;

public:

    Matrix(size_t rows, size_t cols);

    Matrix(const Matrix &other);

    Matrix(size_t rows, size_t cols, const T *dataArray);

    Matrix(size_t rows, size_t cols, Matrix<T>& parentMat, size_t startRows, size_t startCols);

    Matrix<T> hardCopy() const;

    static Matrix<T> zeros(size_t rows, size_t cols);

    Matrix &operator=(const Matrix &other);

    T &operator()(size_t row, size_t col);

    const T &operator()(size_t row, size_t col) const;

    T getValue(size_t row, size_t col) const;

    void setValue(size_t row, size_t col, const T &value);

    Matrix operator+(const Matrix &other) const;

    Matrix operator-(const Matrix &other) const;

    Matrix operator*(const Matrix &other) const;

    Matrix operator+(const T &value) const;

    Matrix operator-(const T &value) const;

    Matrix operator*(const T &value) const;

    Matrix operator/(const T &value) const;

    bool operator==(const Matrix &other) const;

    bool operator!=(const Matrix &other) const;

    ~Matrix();

    Matrix<T> &operator+=(const Matrix &other);

    Matrix<T> &operator-=(const Matrix &other);

    Matrix<T> &operator*=(const Matrix &other);

    Matrix<T> &operator+=(const T &value);

    Matrix<T> &operator-=(const T &value);

    Matrix<T> &operator*=(const T &value);

    Matrix<T> &operator/=(const T &value);

    void print() const;

    void printData() const;

    friend std::ostream &operator
    <<<>(
    std::ostream &os,
    const Matrix<T> &matrix
    );

    friend std::istream &operator>><>(std::istream &is, Matrix<T> &matrix);
};



#include "matrix.cpp"

#endif //  MATRIX_HPP
