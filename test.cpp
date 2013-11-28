#include <RandomForest.h>
#include <feature.h>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <boost/timer.hpp>
#include <util.h>

using namespace std;
using namespace Eigen;

int main(int argc, char *argv[])
{
    vector<vector<double>> X;
    vector<int> y;

    MatrixXd X2,X_test2;
    VectorXi y2, y_test2;

    const string file(argv[1]);
    const int feature_dim = boost::lexical_cast<int>(argv[2]);
    tie(X,y) = readLibsvm<double>(file,feature_dim);
    tie(X2,y2) = readLibsvmEigen<double>(file,feature_dim);

    const int n_classes = countUnique(y);

    boost::timer t;
    const std::function<IdentityFeature* ()> featureFactory = std::bind(createFeature, feature_dim);
    const int n_trees = 1;

    RandomForest<IdentityFeature> forest(n_classes,n_trees);
    forest.train(X, y, featureFactory);
    const string test_file(argv[3]);

    vector<vector<double>> X_test;
    vector<int> y_test;

    tie(X_test,y_test) = readLibsvm<double>(test_file,feature_dim);
    tie(X_test2,y_test2) = readLibsvmEigen<double>(test_file,feature_dim);

    int n_correct = 0;
    const int n_test = y_test2.size();
    vector<int> prediction = forest.predict(X_test2);
    for (int i = 0; i < n_test; ++i)
    {
        std::cout << prediction[i] << std::endl;
       n_correct += (prediction[i] == y_test2(i));
    }

    std::cout << "Accuracy: " << static_cast<double>(n_correct) / n_test * 100 << std::endl;
    std::cout << "Elapsed time: " << t.elapsed() << std::endl;

    return 0;
}
