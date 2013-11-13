#include <memory>
#include <Histogram.h>
#include <vector>

class Node
{
public:

    typedef Node* NodeRawPtr;
    typedef std::unique_ptr<Node> NodePtr;

    Node(int index, bool leaf):
        node_index(index),
        is_leaf(leaf)
    {
    }

    virtual ~Node(){}

    int getNodeIndex() const { return node_index;}

    bool isLeaf() const { return is_leaf;}

private:

    int node_index;
    bool is_leaf;
};

class LeafNode : public Node
{
public:

   LeafNode(int node_index, const Histogram& h):
       Node(node_index, true),
       hist(h),
       dist(hist.getNumberOfBins(),0)
   {
      const int n_samples = hist.getNumberOfSamples();
      const int n_bins = hist.getNumberOfBins();
      for (int i = 0; i < n_bins; ++i) {
         dist[i] = static_cast<double>(hist.getCounts(i)) / n_samples;
      }
   }

   std::vector<double> getDistribution() { return dist;}

private:

   Histogram hist;
   std::vector<double> dist;
};

template <typename FeatureType>
class SplitNode : public Node
{
public:

    typedef Node::NodePtr NodePtr;
    typedef Node::NodeRawPtr NodeRawPtr;

    SplitNode(int node_index_, std::shared_ptr<FeatureType> f, double t)
        : Node(node_index_, false),
          feature(f),
          threshold(t)
    {
    }

    template <typename Data>
    double getFeatureResponse(const Data& v) const
    {
        return (*feature)(v);
    }

    void setLeftChild(NodeRawPtr child)
    {
        left = NodePtr(child);
    }

    void setRightChild(NodeRawPtr child)
    {
        right = NodePtr(child);
    }

    NodeRawPtr getLeftChild() const { return left.get();}

    NodeRawPtr getRightChild() const { return right.get();}

    double getThreshold() const { return threshold;}

private:

    std::shared_ptr<FeatureType> feature;
    double threshold;
    NodePtr left;
    NodePtr right;
};
