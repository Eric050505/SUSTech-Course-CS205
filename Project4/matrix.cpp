#ifndef MATRIX_CPP
#define MATRIX_CPP

// #include "OpenBLAS/cblas.h"

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols): rows(rows), cols(cols), isROI(false), parent(nullptr) {
    if (rows == 0 || cols == 0)
        throw std::invalid_argument("The dimension of the matrix can not be ZERO.");
    data = std::shared_ptr<T[]>(new T[rows * cols], std::default_delete<T[]>());
}

template<typename T>
Matrix<T>::Matrix(const Matrix &other): rows(other.rows), cols(other.cols), isROI(true),
                                        parent((Matrix<int> *) &other) {

}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, Matrix<T> &parentMat, size_t startRows, size_t startCols)
        : rows(rows), cols(cols), isROI(true), parent(&parentMat) {
    if (startRows + rows > parentMat.rows || startCols + cols > parentMat.cols)
        throw std::out_of_range("Sub-matrix dimensions are out of range.");
    data = std::shared_ptr<T[]>(parentMat.data, parentMat.data.get() + startRows * parentMat.cols + startCols);
}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, const T *dataArray): rows(rows), cols(cols), isROI(false), parent(nullptr) {
    if (rows == 0 || cols == 0)
        throw std::invalid_argument("The dimension of the matrix can not be ZERO.");
    data = std::shared_ptr<T[]>(new T[rows * cols], std::default_delete<T[]>());
    std::memcpy(data.get(), dataArray, rows * cols * sizeof(T));
}

template <typename T>
Matrix<T> Matrix<T>::hardCopy() const {
    Matrix<T> copy(rows, cols);
    if (isROI && parent) {
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                copy.data[i * cols + j] = (*this)(i, j);
    } else
        std::memcpy(copy.data.get(), data.get(), rows * cols * sizeof(T));
    return copy;
}

template <typename T>
Matrix<T> Matrix<T>::zeros(size_t rows, size_t cols) {
    Matrix<T> zeroMatrix(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i)
        zeroMatrix.data[i] = 0;
    return zeroMatrix;
}


template<typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix& other) {
    if (this == &other)
        return *this;
    rows = other.rows;
    cols = other.cols;
    isROI = other.isROI;
    parent = other.parent;
    data = other.data;
    return *this;
}



template<typename T>
T &Matrix<T>::operator()(size_t row, size_t col) {
    if (row >= rows || col >= cols)
        throw std::out_of_range("The desired index is out of range. ");
    if (isROI)
        return data[row * parent->cols + col];
    else
        return data[row * cols + col];
}

template<typename T>
const T &Matrix<T>::operator()(size_t row, size_t col) const {
    if (row >= rows || col >= cols)
        throw std::out_of_range("The desired index is out of range. ");
    if (isROI)
        return data[row * parent->cols + col];
    return data[row * cols + col];
}

template<typename T>
void Matrix<T>::setValue(size_t row, size_t col, const T &value) {
    if (row >= rows || col >= cols)
        throw std::out_of_range("The desired index is out of range.");
    if (isROI)
        data[row * parent->cols + col] = value;
    else
        data[row * cols + col] = value;
}

template<typename T>
T Matrix<T>::getValue(size_t row, size_t col) const {
    if (row >= rows || col >= cols)
        throw std::out_of_range("The desired index is out of range.");
    if (isROI)
        return data[row * parent->cols + col];
    else
        return data[row * cols + col];
}

template<typename T>
Matrix<T> Matrix<T>::operator+(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("矩阵维度必须匹配才能相加。");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] + other.data[i];
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("矩阵维度必须匹配才能相减。");
    }
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] - other.data[i];
    }
    return result;
}

/*template<typename T>
Matrix<T> Matrix<T>::operator*(const Matrix& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    Matrix<T> result(rows, other.cols);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                rows, other.cols, cols,
                1.0, data.get(), cols,
                other.data.get(), other.cols,
                0.0, result.data.get(), other.cols);
    return result;
}*/


template<typename T>
Matrix<T> Matrix<T>::operator+(const T &value) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] + value;
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator-(const T &value) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] - value;
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator*(const T &value) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] * value;
    }
    return result;
}

template<typename T>
Matrix<T> Matrix<T>::operator/(const T &value) const {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows * cols; ++i) {
        result.data[i] = data[i] / value;
    }
    return result;
}

template<typename T>
bool Matrix<T>::operator==(const Matrix &other) const {
    if (rows != other.rows || cols != other.cols)
        return false;

    for (size_t i = 0; i < rows * cols; ++i) {
        if (data[i] != other.data[i]) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool Matrix<T>::operator!=(const Matrix &other) const {
    return *this != other;
}

template<typename T>
Matrix<T>::~Matrix() = default;

// << 操作符重载，用于输出矩阵数据
template<typename T>
std::ostream &operator<<(std::ostream &os, const Matrix<T> &matrix) {
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            os << matrix(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}

// >> 操作符重载，用于输入矩阵数据
template<typename T>
std::istream &operator>>(std::istream &is, Matrix<T> &matrix) {
    for (size_t i = 0; i < matrix.rows; ++i) {
        for (size_t j = 0; j < matrix.cols; ++j) {
            is >> matrix(i, j);
        }
    }
    return is;
}

// 新增的操作符重载实现
template<typename T>
Matrix<T> &Matrix<T>::operator+=(const Matrix &other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("The size does not match. ");
    }
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] += other.data[i];
    }
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator-=(const Matrix &other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("The size does not match. ");
    }
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] -= other.data[i];
    }
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator*=(const Matrix &other) {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("The size does not match. ");
    }
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] *= other.data[i];
    }
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator+=(const T &value) {
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] += value;
    }
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator-=(const T &value) {
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] -= value;
    }
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator*=(const T &value) {
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] *= value;
    }
    return *this;
}

template<typename T>
Matrix<T> &Matrix<T>::operator/=(const T &value) {
    for (size_t i = 0; i < rows * cols; ++i) {
        data[i] /= value;
    }
    return *this;
}

template<typename T>
void Matrix<T>::print() const {
    std::cout << "Matrix dimensions: \n" << rows << " * " << cols << std::endl;
    std::cout << "Matrix data type: \n" << typeid(T).name() << std::endl;
    std::cout << "Matrix data: \n" << *this << std::endl;
}

template<typename T>
void Matrix<T>::printData() const {
    std::cout << "Matrix data: \n" << *this << std::endl;
}

#endif