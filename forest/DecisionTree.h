#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <Histogram.h>
#include <util.h>
#include <limits>
#include <queue>
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
          n_nodes(0)
    {

    }

    template <typename FeatureContainer, typename LabelContainer>
    void train(const FeatureContainer& X,
               const LabelContainer& y,
               std::vector<int>& indices,
               const std::function<FeatureType* ()>& factory)
    {
        buildTree(X, y, indices, factory);
    }

    template <typename D>
    std::vector<double> predictDistribution(const D& x) const
    {
        int index = 0;
        while(!nodes[index]->isLeaf())
        {
            SplitNode<FeatureType>* node = dynamic_cast<SplitNode<FeatureType>*>(nodes[index].get());
            const double response = node->getFeatureResponse(x);
            if(response < node->getThreshold()) index = nodes[index]->getLeftChildIndex();
            else index = nodes[index]->getRightChildIndex();
        }
        return dynamic_cast<LeafNode*>(nodes[index].get())->getDistribution();
    }

private:

    struct NodeBuildInfo
    {
        NodeBuildInfo(int from_, int to_, int parent_index_, bool is_left_, int depth_):
            from(from_),
            to(to_),
            parent_index(parent_index_),
            is_left(is_left_),
            depth(depth_)
        {
        }

        const int from;
        const int to;
        const int parent_index;
        const bool is_left;
        const int depth;
    };

    template <typename FeatureContainer,typename LabelContainer>
    void buildTree(const FeatureContainer& X,
                         const LabelContainer& y,
                         std::vector<int>& indices,
                         const std::function<FeatureType* ()>& factory)
    {
        std::queue<NodeBuildInfo> que;
        que.push(NodeBuildInfo(0, indices.size(), 0, false, 0));

        while(!que.empty())
        {
            NodeBuildInfo info(que.front());
            que.pop();

            const int parent_index = info.parent_index;
            const int from = info.from;
            const int to = info.to;
            const bool is_left = info.is_left;
            const int node_index = n_nodes;
            const int n_data = to - from;
            const int depth = info.depth + 1;

            std::vector<double> response(n_data);
            double best_gain = -1;
            std::shared_ptr<FeatureType> best_feature(factory());
            double best_thres;

            std::vector<double> threshold(n_thres_per_feat+1,std::numeric_limits<double>::max());

            Histogram parent_hist(n_classes);
            int prev_label = y[indices[from]];
            bool same_label = true;

            for(int i = from; i < to; ++i)
            {
                int l = y[indices[i]];
                parent_hist.accumulate(l);
                if(l != prev_label) same_label = false;
                prev_label = l;
            }

            if(same_label || n_data <= 5)
            {
                LeafNode* leaf = new LeafNode(node_index, depth, parent_hist);
                nodes.push_back(NodePtr(leaf));
                ++n_nodes;

                if(is_left)
                {
                    nodes[parent_index]->setLeftChildIndex(node_index);
                }
                else
                {
                    nodes[parent_index]->setRightChildIndex(node_index);
                }

                continue;
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
                LeafNode* leaf = new LeafNode(node_index, depth, parent_hist);
                nodes.push_back(NodePtr(leaf));

                ++n_nodes;
                if(is_left)
                {
                    nodes[parent_index]->setLeftChildIndex(node_index);
                }
                else
                {
                    nodes[parent_index]->setRightChildIndex(node_index);
                }

                continue;
            }

            for(int i = from; i < to; ++i)
            {
                response[i-from] = (*best_feature)(X[indices[i]]);
            }

            SplitNode<FeatureType>* split(new SplitNode<FeatureType>(node_index, depth, best_feature, best_thres));
            nodes.push_back(NodePtr(split));
            ++n_nodes;
            if(is_left)
            {
                nodes[parent_index]->setLeftChildIndex(node_index);
            }
            else
            {
                nodes[parent_index]->setRightChildIndex(node_index);
            }

            if(parent_index == 0)
            {
                std::cout << nodes[0]->getLeftChildIndex() << "," << nodes[0]->getRightChildIndex() << std::endl;
            }
            int thres_index = partitionByResponse(indices,from, to, response, best_thres);

            que.push(NodeBuildInfo(from, thres_index, node_index, true, depth));
            que.push(NodeBuildInfo(thres_index, to, node_index, false, depth));

        }
    }

    const int n_classes;
    const int n_candidate_feat;
    const int n_thres_per_feat;
    int n_nodes;
    std::vector<NodePtr> nodes;
};
