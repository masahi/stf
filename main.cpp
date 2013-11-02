#include <RandomForest.h>
#include <feature.h>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <util.h>

using namespace std;

int IdentityFeature::FEATURE_DIM;

int main(int argc, char *argv[])
{
    vector<vector<double>> X;
    vector<int> y;

    const string file(argv[1]);
    int feature_dim = boost::lexical_cast<int>(argv[2]);
    tie(X,y) = readLIBSVM<double>(file,feature_dim);

    const int n_classes = countUnique(y);

    IdentityFeature::FEATURE_DIM = feature_dim;
    RandomForest<IdentityFeature> forest(n_classes);
    forest.train(X, y);

    const string test_file(argv[3]);
    vector<vector<double>> X_test;
    vector<int> y_test;

    tie(X_test,y_test) = readLIBSVM<double>(test_file,feature_dim);

    int n_correct = 0;
    int n_test = y.size();
    for (int i = 0; i < n_test; ++i)
    {
       int c = forest.predict(X[i]);
       std::cout << c << std::endl;
       n_correct += (c == y[i]);
    }

    std::cout << "Accuracy: " << static_cast<double>(n_correct) / n_test * 100 << std::endl;


    return 0;
}
