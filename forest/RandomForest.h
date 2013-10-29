#include <vector>
#include <DecisionTree.h>
#include <util.h>

template <typename FeatureType>
class RandomForest
{
public:

    RandomForest(int n_classes_, int n_trees_ = 100)
        :n_classes(n_classes_),
         n_trees(n_trees_)
    {

    }

    ~RandomForest()
    {

    }

    template <typename D>
    void train(const std::vector<D>& X, const std::vector<int>& y)
    {
        const int data_per_tree = X.size() * 0.25;

        for(auto tree: trees)
        {
            std::vector<int> indices = randomSamples(X.size(), data_per_tree);
            tree->train(X, y, indices);
        }
    }

private:
    const int n_classes;
    const int n_trees;
    std::vector<DecisionTree<FeatureType>*> trees;
};
