#include <RandomForest.h>
#include <feature.h>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <algorithm>
#include <boost/timer.hpp>
#include <util.h>

typedef std::function<IdentityFeature ()> FeatureFactory;

template <>
std::vector<IdentityFeature> generateRandomFeatures(const FeatureFactory& factory, int n)
{
    std::vector<IdentityFeature> features;
    const int feature_dim = factory().getFeatureDim();
    std::vector<int> feature_index(feature_dim);
    std::iota(feature_index.begin(), feature_index.end(),0);

    for (int i = 0; i < n; ++i)
    {
        const int j = randInt(0, feature_dim - i);
        std::swap(feature_index[feature_dim - i - 1], feature_index[j]);
        features.push_back(IdentityFeature(feature_index[feature_dim - i - 1], feature_dim));
    }
    return features;
}

using namespace std;

int main(int argc, char *argv[])
{
    vector<vector<double>> X;
    vector<int> y;

    const string file(argv[1]);
    const int feature_dim = boost::lexical_cast<int>(argv[2]);
    tie(X,y) = readLibsvm<double>(file,feature_dim);

    const int n_classes = countUnique(y);

    boost::timer t;
    const FeatureFactory factory = std::bind(createFeature, feature_dim);
    const int n_trees = 10;
    const int n_features = static_cast<int>(std::sqrt(feature_dim));
    //const int n_thres = -1;

    const std::vector<double> weights(n_classes, 1.0/n_classes);

    RandomForest<IdentityFeature> forest(n_classes, n_trees, n_features);//, n_thres);
    forest.train(X, y, factory, weights);

    const string test_file(argv[3]);

    vector<vector<double>> X_test;
    vector<int> y_test;

    tie(X_test,y_test) = readLibsvm<double>(test_file,feature_dim);

    int n_correct = 0;
	const int n_test = y_test.size();
    std::vector<int> pred_count(n_classes, 0);
    for (int i = 0; i < n_test; ++i)
    {
		const int prediction = forest.predict(X_test[i]);
		n_correct += (prediction == y_test[i] ? 1 : 0);
        pred_count[prediction] += 1;
    }

    std::cout << "Accuracy: " << static_cast<double>(n_correct) / n_test * 100 << std::endl;
    std::cout << "Elapsed time: " << t.elapsed() << std::endl;

    return 0;
}
