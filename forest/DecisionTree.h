#include <vector>
#include <Eigen/Core>
#include <Histogram.h>

template <typename FeatureType>
class Node
{
public:

    Node(const FeatureType& f, double t, bool l)
        : feature(f),
          threshold(t),
          isLeaf(l)
    {

    }

    template <typename Data>
    double getFeatureResponse(Data v) const
    {
        return feature(v);
    }

    void addSample(int label)
    {
        hist.accumulate(label);
    }

private:
    const FeatureType& feature;
    const double threshold;
    bool isLeaf;
    Histogram hist;
    Node *left;
    Node *right;
};

template <typename FeatureType>
class DecisionTree
{
public:
    typedef Eigen::MatrixXf MatrixType;
    typedef Eigen::VectorXf VectorType;
    typedef int LabelType;
    typedef Eigen::VectorXi LabelVector;

    DecisionTree(int n_classes_)
        : n_classes(n_classes_)
    {
    }

    ~DecisionTree()
    {

    }

    template <typename DataVector, typename LabelVector>
    void train(const DataVector& X, const LabelVector& y)
    {
        root = buildTree(X, y, 0, y.size());
    }


private:

    template <typename DataVector, typename LabelVector>
    Node* buildTree(const DataVector& X, const LabelVector& y, int from, int to)
    {

    }

    const int nClasses;
    const int nNodes;
    Node<FeatureType> *root;
};
