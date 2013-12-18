#include <memory>
#include <algorithm>
#include <Histogram.h>
#include <vector>

class Node
{
public:

    typedef Node* NodeRawPtr;
    typedef std::shared_ptr<Node> NodePtr;

    Node(int index, bool leaf, int d):
        node_index(index),
        is_leaf(leaf),
        depth(d),
        left_child_index(-1),
        right_child_index(-1)
    {
    }

    virtual ~Node(){}

    void setLeftChildIndex(int index)
    {
        left_child_index = index;
    }

    void setRightChildIndex(int index)
    {
        right_child_index = index;
    }

    int getLeftChildIndex() const { return left_child_index;}
    int getRightChildIndex() const { return right_child_index;}
    int getNodeIndex() const { return node_index;}
    bool isLeaf() const { return is_leaf;}

private:

    const int node_index;
    const bool is_leaf;
    const int depth;
    int left_child_index;
    int right_child_index;
};

class LeafNode : public Node
{
public:

   LeafNode(int node_index, int depth, const Histogram& h):
       Node(node_index, true, depth),
       hist(h),
	   dist(h.getNumberOfBins(),0)
   {
      const double n_samples = hist.getNumberOfSamples();
      const int n_bins = hist.getNumberOfBins();
      const std::vector<double> bikns = hist.getBins();
      for (int i = 0; i < n_bins; ++i) {
         dist[i] = static_cast<double>(hist.getCounts(i)) / n_samples;
      }
   }

   const std::vector<double>& getDistribution() const { return dist;}

private:

   const Histogram& hist;
   std::vector<double> dist;
};

template <typename FeatureType>
class SplitNode : public Node
{
public:

    typedef Node::NodePtr NodePtr;
    typedef Node::NodeRawPtr NodeRawPtr;

    SplitNode(int node_index, int depth, std::shared_ptr<FeatureType> f, double t)
        : Node(node_index, false, depth),
          feature(f),
          threshold(t)
    {
    }

    template <typename Data>
    double getFeatureResponse(const Data& v) const
    {
        return (*feature)(v);
    }

    double getThreshold() const { return threshold;}

private:

    std::shared_ptr<FeatureType> feature;
    double threshold;
};
