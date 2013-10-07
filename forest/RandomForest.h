#include <vector>
#include <boost/foreach.hpp>
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

private:
    const int n_classes;
    const int n_trees;
    std::vector<DecisionTree<FeatureType> *> trees;
};
