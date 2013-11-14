#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <Histogram.h>
#include <util.h>
#include <limits>
#include <Node.h>


template <typename FeatureType>
class DecisionTree
{
public:

    typedef Node::NodePtr NodePtr;
    typedef Node::NodeRawPtr NodeRawPtr;

    DecisionTree(int n_classes_)
        : n_classes(n_classes_),
          n_candidate_feat(10),
          n_thres_per_feat(100),
          n_nodes(0),
          root(nullptr)
    {

    }

    template <typename FeatureContainer, typename LabelContainer>
    void train(const FeatureContainer& X,
               const LabelContainer& y,
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
            SplitNode<FeatureType>* node = dynamic_cast<SplitNode<FeatureType>*>(current_node);
            double response = node->getFeatureResponse(x);
            if(response < node->getThreshold()) current_node = node->getLeftChild();
            else current_node = node->getRightChild();
        }
        return dynamic_cast<LeafNode*>(current_node)->getDistribution();
    }

private:

    template <typename FeatureContainer,typename LabelContainer>
    NodeRawPtr buildTree(const FeatureContainer& X,
                         const LabelContainer& y,
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
                LeafNode* leaf = new LeafNode(n_nodes++, parent_hist);
                return leaf;
            }
        }

        if(n_data <=5 )
        {
            LeafNode* leaf = new LeafNode(n_nodes++, parent_hist);
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
            LeafNode* leaf = new LeafNode(n_nodes++,parent_hist);
            return leaf;
        }

        for(int i = from; i < to; ++i)
        {
            response[i-from] = (*best_feature)(X[indices[i]]);
        }

        SplitNode<FeatureType>* parent(new SplitNode<FeatureType>(n_nodes++,best_feature, best_thres));

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
