#include <vector>
#include <DecisionTree.h>
#include <memory>
#include <iostream>
#include <util.h>
#include <algorithm>
#include <functional>
#include <numeric>

template <typename FeatureType>
class RandomForest
{
public:

    typedef DecisionTree<FeatureType> Tree;
    typedef std::unique_ptr<Tree> TreePtr;

    RandomForest(int n_classes_, int n_trees_ = 1)
        :n_classes(n_classes_),
         n_trees(n_trees_)
         //trees(n_trees, std::move(TreePtr(new Tree(n_classes))))
    {
        for(int i = 0; i < n_trees; ++i) trees.push_back(std::move(TreePtr(new Tree(n_classes))));
    }

    ~RandomForest()
    {
    }

    template <typename D>
    void train(const std::vector<D>& X, const std::vector<int>& y)
    {
        const int data_per_tree = X.size() * 0.25;
        for(int i = 0; i < n_trees; ++i)
        {
           // std::vector<int> indices = randomSamples(X.size(), data_per_tree);
            std::vector<int> indices(X.size());
            std::iota(indices.begin(), indices.end(), 0);
            std::random_shuffle(indices.begin(), indices.end());
            trees[i]->train(X, y, indices);
        }
    }

    template <typename D>
    std::vector<double> predictDistribution(const D& x)
    {
        std::vector<double> dist(n_classes, 0);
        for (int i = 0; i < n_trees; ++i)
        {
           std::vector<double> tree_dist = trees[i]->predictDistribution(x);
           std::transform(dist.begin(), dist.end(), tree_dist.begin(), dist.begin(), std::plus<double>());

        }

        using std::placeholders::_1;
        std::transform(dist.begin(), dist.end(), dist.begin(), std::bind(std::divides<double>(), _1,n_trees));
        return dist;
    }

    template <typename D>
    int predict(const D& x)
    {
        std::vector<double> dist = predictDistribution(x);
        return std::max_element(dist.begin(), dist.end()) - dist.begin();
    }

private:
    const int n_classes;
    const int n_trees;
    std::vector<TreePtr> trees;
};
