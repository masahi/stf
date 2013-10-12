#include <RandomForest.h>
#include <feature.h>
#include <memory>

using namespace std;


int main(int argc, char *argv[])
{
  //   vector<vector<double> > X;
  //    vector<int> y;

    const int n_classes = 2;
    // IdentityFeature::FEATURE_DIM = 20;
    RandomForest<IdentityFeature> forest(n_classes);
    // forest.train();

    return 0;
}
