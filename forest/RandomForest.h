#include <vector>
#include <DecisionTree.h>
#include <memory>
#include <iostream>
#include <util.h>
#include <algorithm>
#include <functional>
#include <numeric>
#include <omp.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

template <typename FeatureType>
class RandomForest
{
public:

    typedef DecisionTree<FeatureType> Tree;
    typedef std::shared_ptr<Tree> TreePtr;

    RandomForest(int n_classes_, int n_trees_ = 1)
        :n_classes(n_classes_),
          n_trees(n_trees_),
          trees(n_trees, TreePtr(new Tree(n_classes)))
    {
    }

    template <typename T>
    void train(const Matrix<T>& X,
               const Vector<int>& y,
               const std::function<FeatureType* ()>& factory)
    {
        train(EigenMatrixAdapter<T>(X), EigenVectorAdapter<int>(y), factory);
    }

    template <typename FeatureContainer, typename LabelContainer>
    void train(const FeatureContainer& X,
               const LabelContainer& y,
               const std::function<FeatureType* ()>& factory)
    {
        const int data_per_tree = X.size();
        //        for(int i = 0; i < n_trees; ++i)
        //        {
        //            std::vector<int> indices = randomSamples(X.size(), data_per_tree);
        //            trees[i]->train(X,y, indices, factory);
        //        }
        tbb::parallel_for(0,
                          n_trees,
                          [&](int i)
        {
            std::vector<int> indices = randomSamples(X.size(), data_per_tree);
            trees[i]->train(X, y, indices, factory);
        }
        );
    }


    template <typename D>
    std::vector<double> predictDistribution(const D& x)
    {
        std::vector<double> zeros(n_classes, 0);

        std::vector<double> unnormalized =
                tbb::parallel_reduce(tbb::blocked_range<int>(0, n_trees),
                                     zeros,
                                     [&](const tbb::blocked_range<int>& range, std::vector<double> init)
        {
            for (int i = range.begin(); i < range.end(); ++i)
            {
                std::vector<double> dist = trees[i]->predictDistribution(x);
                init += dist;
            }
            return init;
        },
        [](std::vector<double> a, std::vector<double> b){ return a + b; }
                );

        return 1.0/n_trees * unnormalized;
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
    std::vector<TreePtr> trees;
};
