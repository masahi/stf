#include <cmath>
#include <vector>
#include <map>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <util/eigen.h>

const double PI = 3.141592;
const double THRESHOLD = 0.01;
const double EPSILON = 0.0000001;

template <typename T>
Matrix<T> logDensity(const Matrix<T>& X, const Matrix<T>& mean, const Matrix<T>& cov)
{
    const int n = X.rows();
    const int dim = X.cols();
    const Vector<T> logdetCov = cov.array().log().rowwise().sum();
    const Matrix<T> invCov = cov.array().inverse();
    const Vector<T> tmp1 = (mean.array().square() * invCov.array()).rowwise().sum();
    const Matrix<T> tmp2 = -2 * X * mean.cwiseProduct(invCov).transpose();
    const Matrix<T> tmp3 = X.array().square().matrix() * invCov.transpose();
    const Matrix<T> tmp4 = (tmp2 + tmp3).rowwise() + (tmp1 + logdetCov).transpose();
    return -0.5 * (dim * std::log(2 * PI) + tmp4.array());

    /*
    No temporary seems slower
    Matrix invCov = cov.array().inverse();
    return -0.5 * (dim * std::log(2 * PI) +
    (
    (-2 * X * mean.cwiseProduct(invCov).transpose() + X.array().square().matrix() * invCov.transpose()).rowwise() +
    ((mean.array().square() * invCov.array()).rowwise().sum() + cov.array().log().rowwise().sum()).transpose().matrix()
    )
    .array());*/
}

template <typename T>
Vector<T> logsumexp(const Matrix<T>& X)
{
    const Matrix<T> arr = X.transpose();
    const Vector<T> vmax = arr.colwise().maxCoeff();
    return (arr.rowwise() - vmax.transpose()).array().exp().colwise().sum().log().transpose().matrix() + vmax;
}

template <typename T>
void scale(Matrix<T>& X, Vector<T>& mean, Vector<T>& std_div)
{
    const int n = X.rows();
    mean = X.colwise().mean();
    std_div = (X.array().square().colwise().sum() / n - mean.transpose().array().square()).sqrt();
    X.rowwise() -= mean.transpose();
    X.array().rowwise() /= std_div.transpose().array();
}

template<typename T = double>
class GMM
{

public:
    GMM(int n_component = 2) :
        K(n_component)
    {
    }

    void train(const Matrix<T>& data, int max_iter = 1000)
    {
        X = data;
        N = X.rows();
        dim = X.cols();
        means = Matrix<T>::Random(K, dim);
        covs = Matrix<T>::Constant(K, dim, 1.0);
        weights = Vector<T>::Constant(K, 1.0 / K);
        scale(X, feature_mean, feature_std_div);
        
        Vector<T> ll, ll_new;
        Matrix<T> gamma;
        estep(ll, gamma);
        
        for (int n = 0; n < max_iter; ++n){
            estep(ll_new, gamma);
            mstep(gamma);
            const float diff = (ll_new - ll).norm();
            const float ll_norm = ll_new.norm();
            //std::cout << ll_norm << std::endl;
            if (n > 0 && diff < THRESHOLD) {
             //   std::cout << "Converged in " << n + 1 << " iteration.\n";
                break;
            }
            ll = ll_new;
        }
    }

    template<typename VectorType>
    double evaluate(const VectorType& x) const
    {
        const Matrix<T> v = ((x - feature_mean).array() / feature_std_div.array()).transpose();
        const Vector<T> density = logDensity(v, means, covs).transpose();
        return (density.array() + weights.array().log()).sum();
    }

    Matrix<T> getMeans() const
    {
        return means;
    }

private:

    void estep(Vector<T>& logprob, Matrix<T>& gamma)
    {
        gamma = (logDensity(X, means, covs)).rowwise() + weights.transpose().array().log().matrix();
        logprob = logsumexp(gamma);
        gamma.array().colwise() -= logprob.array();
        gamma = gamma.array().exp();
    }

    void mstep(const Matrix<T>& gamma)
    {
        const Vector<T> w = gamma.colwise().sum();
        const Matrix<T> weighted_X = gamma.transpose() * X;
        const Vector<T> inverse_w = (w.array() + EPSILON).inverse();
        weights = (1.0 / (w.sum() + EPSILON)) * w;
        means = weighted_X.array().colwise() * inverse_w.array();
        const Matrix<T> second_moment = (gamma.transpose() * X.array().square().matrix()).array().colwise() * inverse_w.array();
        covs = second_moment.array() - means.array().square();
    }

    const int K;
    int N;
    int dim;
    Matrix<T> X;
    Matrix<T> means;
    Matrix<T> covs;
    Vector<T> weights;
    Vector<T> feature_mean;
    Vector<T> feature_std_div;
};


