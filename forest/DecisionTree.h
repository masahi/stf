#include <vector>
#include <Histogram.h>

template <typename FeatureType>
class Node
{
public:
    Node(const FeatureType& f, double t, bool l, const Histogram& h)
        : feature(f),
          threshold(t),
          is_leaf(l),
          hist(h)
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

    void setLeftChild(Node* child)
    {
        left = child
    }

    void setRightChild(Node* child)
    {
        right = child
    }

private:
    const FeatureType& feature;
    const double threshold;
    bool is_leaf;
    Histogram hist;
    Node* left;
    Node* right;
};

template <typename FeatureType>
class DecisionTree
{
public:


    DecisionTree(int n_classes_)
        : n_classes(n_classes_)
    {
    }

    ~DecisionTree()
    {

    }

    template <typename D>
    void train(const std::vector<D>& X, const std::vector<int>& y)
    {
        root = buildTree(X, y, 0, y.size());
    }


private:

    template <typename D>
    Node* buildTree(const std::vector<D>& X, const std::vector<int>& y, int from, int to)
    {
        //find best feature and thres
        Node* parent = new Node();
        int thres_index;
        //recurese on left and right child
        Node* l_child = buildTree(X, y, from, thres_index);
        Node* r_child = buildTree(X, y, thres_index, to);
        parent->setLeftChild(l_child);
        parent->setRightChild(r_child);
        return parent;
    }

    const int nClasses;
    const int nNodes;
    Node<FeatureType> *root;
};
