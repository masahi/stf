#include <vector>
#include <DecisionTree.h>

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

        for(auto tree: trees)
        {
            std::vector<int> indices = random_samples(X.size());
            tree->train(X, y, indices);
        }
    }

private:
    const int n_classes;
    const int n_trees;
    std::vector<DecisionTree<FeatureType>*> trees;
};
