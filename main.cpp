#include <RandomForest.h>
#include <feature.h>
#include <memory>
#include <iostream>

using namespace std;

int IdentityFeature::FEATURE_DIM;

int main(int argc, char *argv[])
{
    vector<vector<double>> X(100,vector<double>(10, 0));
    vector<int> y(100,0);

    const int n_classes = 2;
    IdentityFeature::FEATURE_DIM = 10;
    RandomForest<IdentityFeature> forest(n_classes);
    forest.train(X, y);

    return 0;
}
