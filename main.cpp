#include <RandomForest.h>
#include <feature.h>
#include <memory>
#include <cstdlib>
#include <iostream>

using namespace std;

int IdentityFeature::FEATURE_DIM;

int main(int argc, char *argv[])
{
    vector<vector<double>> X(10,vector<double>(5, 0));
    vector<int> y(10,0);

    for(int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 5; ++j) {
            X[i][j] = (double)rand() / RAND_MAX;
        }
       if(i%2) y[i] = 1;
    }
    const int n_classes = 2;
    IdentityFeature::FEATURE_DIM = 5;
    RandomForest<IdentityFeature> forest(n_classes);
    forest.train(X, y);

    return 0;
}
