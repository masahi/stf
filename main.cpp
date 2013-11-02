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

    return 0;
}
