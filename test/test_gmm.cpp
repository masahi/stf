#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/timer.hpp>
#include <Eigen/Core>
#include <gmm/GMM.h>

typedef float T;

int main(int argc, char *argv[])
{
    const Matrix<T> X = Matrix<T>::Random(100000,10);
    const int n_component = 5;
    GMM<T> gmm(n_component);
    boost::timer t;
    gmm.train(X);
    std::cout << t.elapsed() << std::endl;
    return 0;
}
