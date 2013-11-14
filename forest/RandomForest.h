#include <vector>
#include <DecisionTree.h>
#include <memory>
#include <iostream>
#include <util.h>
#include <algorithm>
#include <functional>
#include <numeric>
#include <omp.h>

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
               const std::function<FeatureType* ()>& factory,
               int n_threads = 1)
    {
        train(EigenMatrixAdapter<T>(X), EigenVectorAdapter<int>(y), factory, n_threads);
    }

    template <typename FeatureContainer,typename LabelContainer>
    void train(const FeatureContainer& X,
               const LabelContainer& y,
               const std::function<FeatureType* ()>& factory,
               int n_threads = 1)
    {
        const int data_per_tree = X.size() * 0.25;
        omp_set_num_threads(n_threads);

#pragma omp parallel for
        for(int i = 0; i < n_trees; ++i)
        {
            std::vector<int> indices = randomSamples(X.size(), data_per_tree);
            trees[i]->train(X, y, indices, factory);
        }
    }


    template <typename D>
    std::vector<double> predictDistribution(const D& x)
    {
        std::vector<double> dist(n_classes, 0);

#pragma omp parallel for
        for (int i = 0; i < n_trees; ++i)
        {
           std::vector<double> tree_dist = trees[i]->predictDistribution(x);
#pragma omp critical
           std::transform(dist.begin(), dist.end(), tree_dist.begin(), dist.begin(), std::plus<double>());
        }

        using std::placeholders::_1;
        std::transform(dist.begin(), dist.end(), dist.begin(), std::bind(std::divides<double>(), _1,n_trees));
        return dist;
    }

    template <typename D>
    int predict(const D& x, int n_threads = 1)
    {
        omp_set_num_threads(n_threads);
        std::vector<double> dist = predictDistribution(x);
        return std::max_element(dist.begin(), dist.end()) - dist.begin();
    }

private:
    const int n_classes;
    const int n_trees;
    std::vector<TreePtr> trees;
};
