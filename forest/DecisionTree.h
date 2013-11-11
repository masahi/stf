#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <Histogram.h>
#include <util.h>
#include <limits>

template <typename FeatureType>
class Node
{
public:
    typedef Node<FeatureType>* NodeRawPtr;
    typedef std::unique_ptr<Node<FeatureType>> NodePtr;

    Node(int node_index_, std::shared_ptr<FeatureType> dummy, const Histogram& h)
        : node_index(node_index_),
          feature(dummy),
          threshold(0),
          hist(h),
          dist(hist.getNumberOfBins(), 0),
          is_leaf(true),
          left(nullptr),
          right(nullptr)
    {
        const int n_samples = hist.getNumberOfSamples();

        std::vector<int> bins = hist.getBins();
      //  std::transform(dist.begin(), dist.end(), bins.begin(), [](int count){ return static_cast<double>(count)/n_samples;});
        for (int i = 0; i < bins.size(); ++i) {
            dist[i] = (double)bins[i] / n_samples;
        }
    }

    Node(int node_index_, std::shared_ptr<FeatureType> f, double t,const Histogram& h)
        : node_index(node_index_),
          feature(f),
          threshold(t),
          hist(h),
          is_leaf(false)
    {
    }

    template <typename Data>
    double getFeatureResponse(Data v) const
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

    int getNodeIndex() const {return node_index;}
    NodeRawPtr getLeftChild() const { return left.get();}
    NodeRawPtr getRightChild() const { return right.get();}
    double getThreshold() const { return threshold;}
    FeatureType getFeature() const { return feature;}
    std::vector<double> getDistribution() const { return dist;}
    bool isLeaf() const { return is_leaf;}

private:
    std::shared_ptr<FeatureType> feature;
    double threshold;
    int node_index;
    Histogram hist;
    std::vector<double> dist;
    bool is_leaf;
    NodePtr left;
    NodePtr right;
};

template <typename FeatureType>
class DecisionTree
{
public:

    typedef typename Node<FeatureType>::NodeRawPtr NodeRawPtr;
    typedef typename Node<FeatureType>::NodePtr NodePtr;

    DecisionTree(int n_classes_)
        : n_classes(n_classes_),
          n_candidate_feat(10),
          n_thres_per_feat(100),
          n_nodes(0),
          root(nullptr)
    {

    }

    template <typename D>
    void train(const std::vector<D>& X,
               const std::vector<int>& y,
               std::vector<int>& indices,
               const std::function<FeatureType* ()>& factory)
    {
        root = NodePtr(buildTree(X, y, indices, 0, y.size(), factory));
    }

    template <typename D>
    std::vector<double> predictDistribution(const D& x)
    {
        NodeRawPtr current_node = root.get();

        while(!current_node->isLeaf())
        {
            double response = current_node->getFeatureResponse(x);
            if(response < current_node->getThreshold()) current_node = current_node->getLeftChild();
            else current_node = current_node->getRightChild();
        }
        return current_node->getDistribution();
    }

private:

    template <typename D>
    NodeRawPtr buildTree(const std::vector<D>& X,
                         const std::vector<int>& y,
                         std::vector<int>& indices,
                         int from, int to,
                         const std::function<FeatureType* ()>& factory)
    {
        const int n_data = to - from;
        std::vector<double> response(n_data);

        double best_gain = -1;
        std::shared_ptr<FeatureType> best_feature(factory());
        double best_thres;

        std::vector<double> threshold(n_thres_per_feat+1,std::numeric_limits<double>::max());

        Histogram parent_hist(n_classes);
        for(int i = from; i < to; ++i)
        {
            int l = y[indices[i]];
            parent_hist.accumulate(l);
        }

        for(int i = 0; i < n_classes; ++i)
        {
            if(parent_hist.getCounts(i) == n_data)
            {
                NodeRawPtr leaf(new Node<FeatureType>(n_nodes++, best_feature,parent_hist));
                return leaf;
            }
        }

        if(n_data <=5 )
        {
            NodeRawPtr leaf(new Node<FeatureType>(n_nodes++, best_feature, parent_hist));
            return leaf;
        }


        int n_threshold;
        for(int i = 0; i < n_candidate_feat; ++i)
        {
            std::shared_ptr<FeatureType> f(factory());
            for(int j = from; j < to; ++j)
            {
                response[j-from] = (*f)(X[indices[j]]);
            }
            if(n_data > n_thres_per_feat)
            {
                for(int j = 0; j < n_thres_per_feat+1; ++j)
                {
                    threshold[j] = response[randInt(0, n_data)];
                    n_threshold = n_thres_per_feat;
                }
            }
            else
            {
                std::copy(response.begin(), response.begin()+n_data, threshold.begin());
                n_threshold = n_data - 1;
            }

            std::sort(threshold.begin(), threshold.begin()+n_threshold);

            if(threshold[0] == threshold[n_threshold-1])
            {
                continue;
            }

            for(int j = 0; j < n_threshold; ++j)
            {
                threshold[j] = threshold[j] + (double)rand()/RAND_MAX * (threshold[j+1] - threshold[j]);
            }

            std::vector<Histogram> partition_statistics(n_threshold+1, Histogram(n_classes));
            for(int j = from; j < to; ++j)
            {
                int t = std::upper_bound(threshold.begin(), threshold.begin()+n_threshold, response[j-from]) - threshold.begin();
                partition_statistics[t].accumulate(y[indices[j]]);
            }

            Histogram left_statistics(n_classes), right_statistics(n_classes);
            left_statistics.accumulate(partition_statistics[0]);

            for (int t = 1; t < n_threshold+1; ++t)
            {
                right_statistics.accumulate(partition_statistics[t]);
            }

            double gain = computeInfomationGain(parent_hist, left_statistics, right_statistics);
            if(gain > best_gain)
            {
                best_gain = gain;
                best_feature = f;
                best_thres = threshold[0];

            }

            for(int t = 1; t < n_threshold; ++t)
            {
                left_statistics.accumulate(partition_statistics[t]);
                right_statistics.decrease(partition_statistics[t]);

                gain = computeInfomationGain(parent_hist, left_statistics, right_statistics);
                // std::cout << t <<"," << gain << std::endl;
                if(gain > best_gain)
                {
                    best_gain = gain;
                    best_feature = f;
                    best_thres = threshold[t];

                }
            }
        }

        if(best_gain <= 0.01)
        {
            NodeRawPtr leaf(new Node<FeatureType>(n_nodes++,best_feature,parent_hist));
            return leaf;
        }

        for(int i = from; i < to; ++i)
        {
            response[i-from] = (*best_feature)(X[indices[i]]);
        }

        NodeRawPtr parent(new Node<FeatureType>(n_nodes++,best_feature, best_thres, parent_hist));

        int thres_index = partitionByResponse(indices,from, to, response, best_thres);

        NodeRawPtr l_child = buildTree(X, y, indices, from, thres_index, factory);
        NodeRawPtr r_child = buildTree(X, y, indices, thres_index, to, factory);
        parent->setLeftChild(l_child);
        parent->setRightChild(r_child);
        return parent;
    }

    const int n_classes;
    const int n_candidate_feat;
    const int n_thres_per_feat;
    int n_nodes;
    NodePtr root;
};
