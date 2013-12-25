#ifndef EIGEN_H
#define EIGEN_H

#include <Eigen/Core>

template <typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

template <typename T>
using VectorMapper = Eigen::Map<Vector<T>>;

template <typename T>
using ConstVectorMapper = Eigen::Map<const Vector<T>>;

template <typename T>
class EigenMatrixAdapter
{
public:

    EigenMatrixAdapter(const Matrix<T>& mat) :
        matrix(mat)
    {
    }

    const size_t size() const
    {
        return matrix.rows();
    }

    Vector<T> operator[](int index) const
    {
        return matrix.row(index).transpose();
    }

private:

    const Matrix<T>& matrix;
};

template <typename T>
class EigenVectorAdapter
{
public:

    EigenVectorAdapter(const Vector<T>& vec) :
        vector(vec)
    {
    }

    const size_t size() const
    {
        return vector.rows();
    }

    T operator[](int index) const
    {
        return vector(index);
    }

private:

    const Vector<T>& vector;
};

#endif // EIGEN_H
