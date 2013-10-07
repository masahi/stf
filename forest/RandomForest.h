#include <Eigen/Core>
#include <vector>
#include <boost/foreach.hpp>
#include <DecisionTree.h>

class RandomForest
{
public:
    typedef DecisionTree::MatrixType MatrixType;
    typedef DecisionTree::VectorType VectorType;
    typedef DecisionTree::LabelType LabelType;
    typedef DecisionTree::LabelVector LabelVector;

    RandomForest(int n_classes_, int n_trees_ = 100):n_classes(n_classes_), n_trees(n_trees_)
    {
        trees.resize(n_trees);
        BOOST_FOREACH(DecisionTree *tree, trees)
        {
            tree = new DecisionTree(n_classes);
        }
    }

    ~RandomForest()
    {
        BOOST_FOREACH(DecisionTree *tree, trees)
        {
            delete tree;
        }
    }

    void train(MatrixType X, VectorType y)
    {
        BOOST_FOREACH(DecisionTree *tree, trees)
        {
            tree->train(X,y);
        }
    }

    LabelVector predict(MatrixType X)
    {
        LabelVector prediction(X.rows());

        for(int i = 0; i < X.rows(); ++i)
        {
            prediction(i) = predict(X.row(i).transpose());
        }

        return prediction;
    }

    LabelType predict(VectorType x)
    {
        MatrixType tree_probs(n_trees,n_classes);

        for(int i = 0; i < n_trees; ++i)
        {
            tree_probs.row(i) = trees[i]->predict_prob(x).transpose();
        }

        LabelType c;
        tree_probs.colwise().mean().maxCoeff(&c);
        return c;
    }



private:
    const int n_classes;
    const int n_trees;
    std::vector<DecisionTree *> trees;
};
