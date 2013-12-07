#include <RandomForest.h>
#include <feature.h>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <boost/timer.hpp>
#include <util.h>

using namespace std;
//using namespace Eigen;

int main(int argc, char *argv[])
{
    vector<vector<double>> X;
    vector<int> y;

    const string file(argv[1]);
    const int feature_dim = boost::lexical_cast<int>(argv[2]);
    tie(X,y) = readLibsvm<double>(file,feature_dim);

    const int n_classes = countUnique(y);

    boost::timer t;
    const std::function<IdentityFeature* ()> featureFactory = std::bind(createFeature, feature_dim);
    const int n_trees = 10;
    const int n_features = static_cast<int>(std::sqrt(feature_dim));
    const int n_thres = -1;

    RandomForest<IdentityFeature> forest(n_classes,n_trees, n_features, n_thres);
    forest.train(X, y, featureFactory);
    const string test_file(argv[3]);

    vector<vector<double>> X_test;
    vector<int> y_test;

    tie(X_test,y_test) = readLibsvm<double>(test_file,feature_dim);

    int n_correct = 0;
	const int n_test = y_test.size();
    for (int i = 0; i < n_test; ++i)
    {
		const int prediction = forest.predict(X_test[i]);
		n_correct += (prediction == y_test[i] ? 1 : 0);
    }

    std::cout << "Accuracy: " << static_cast<double>(n_correct) / n_test * 100 << std::endl;
    std::cout << "Elapsed time: " << t.elapsed() << std::endl;

    return 0;
}
