#include <vector>
#include <DecisionTree.h>
#include <memory>
#include <iostream>
#include <util.h>
#include <algorithm>
#include <functional>
#include <numeric>
#include <omp.h>
#include <limits>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

template <typename FeatureType>
class RandomForest
{
public:

    typedef DecisionTree<FeatureType> Tree;

    RandomForest(int n_classes_, int n_trees_ = 1, int n_features = 400, int n_thresholds = -1, int max_depth = std::numeric_limits<int>::max())
        :n_classes(n_classes_),
        n_trees(n_trees_),
        trees(n_trees, Tree(n_classes, n_features, n_thresholds, max_depth))
    {
    }

    template <typename T>
    void train(const Matrix<T>& X,
        const Vector<int>& y,
        const std::function<FeatureType ()>& factory)
    {
        train(EigenMatrixAdapter<T>(X), EigenVectorAdapter<int>(y), factory);
    }

    template <typename FeatureContainer, typename LabelContainer>
    void train(const FeatureContainer& X,
        const LabelContainer& y,
        const std::function<FeatureType ()>& factory,
        const std::vector<double>& class_weights)
    {
        const int data_per_tree = X.size();

        tbb::parallel_for(0,
            n_trees,
            [&](int i)
        {
            std::vector<int> indices = randomSamples(X.size(), data_per_tree);
            trees[i].train(X, y, indices, factory,class_weights);
        }
        );
    }

    template <typename D>
    std::vector<double> predictDistribution(const D &x) {
        std::vector<double> zeros(n_classes, 0);

        std::vector<double> unnormalized =
            tbb::parallel_reduce(tbb::blocked_range<int>(0, n_trees),
            zeros,
            [&](const tbb::blocked_range<int>& range, std::vector<double> init)
        {
            for (int i = range.begin(); i < range.end(); ++i)
            {
                std::vector<double> dist = trees[i].predictDistribution(x);
                init += dist;
            }
            return init;
        },
            [](const std::vector<double>& a, const std::vector<double>& b){ return a + b; }
        );

        return 1.0 / n_trees * unnormalized;
    }

    template <typename T>
    std::vector<int> predict(const Matrix<T>& X)
    {
        EigenMatrixAdapter<T> adapter(X);
        const int n_samples = adapter.size();
        std::vector<int> prediction(n_samples);

        for (int i = 0; i < n_samples; ++i)
        {
            prediction[i] = predict(adapter[i]);
        }

        return prediction;
    }

    template <typename D>
    int predict(const D& x)
    {
        std::vector<double> dist = predictDistribution(x);
        return argmax(dist);
    }

private:

    const int n_classes;
    const int n_trees;
    std::vector<Tree> trees;
};
