#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <functional>
#include <limits>
#include <queue>
#include <forest/Histogram.h>
#include <forest/Node.h>
#include <util/general.h>

template <typename FeatureType>
std::vector<FeatureType> generateRandomFeatures(const std::function<FeatureType()>& factory, int n)
{
    std::vector<FeatureType> features(n, factory());
    std::generate(features.begin(), features.end(), factory);
    return features;
}

void generateCandidateThreshold(std::vector<double>& threshold, const std::vector<double>& response, int n_threshold, int n_data)
{
    if (n_data - 1 != n_threshold)
    {
        for (int j = 0; j < threshold.size(); ++j)
        {
            threshold[j] = response[randInt(0, n_data)];
        }
    }
    else
    {
        std::copy(response.begin(), response.end(), threshold.begin());
    }

    std::sort(threshold.begin(), threshold.end());

//    if (threshold[0] == threshold[n_threshold])
//    {
//        continue;
//    }

    for (int j = 0; j < n_threshold; ++j)
    {
        threshold[j] = threshold[j] + (double)rand() / RAND_MAX * (threshold[j + 1] - threshold[j]);
    }
    //    if (n_data == n_threshold + 1)
    //    {
    //        std::copy(response.begin(), response.end(), threshold.begin());
    //    }
    //    else
    //    {
    //        for (int j = 0; j < threshold.size(); ++j)
    //        {
    //            threshold[j] = response[randInt(0, n_data)];
    //        }
    //    }

    //    std::sort(threshold.begin(), threshold.end());

    //    for (int j = 0; j < n_threshold; ++j)
    //    {
    //        threshold[j] = threshold[j] + static_cast<double>(rand()) / RAND_MAX * (threshold[j + 1] - threshold[j]);
    //    }
}

double computeInfomationGain(const Histogram& parent, const Histogram& left, const Histogram& right)
{
	const int n_classes = parent.getNumberOfBins();
	std::vector<double> parent_prob(n_classes, 0);
	std::vector<double> left_prob(n_classes, 0);
	std::vector<double> right_prob(n_classes, 0);
	double parent_entoropy = 0;
	for (int i = 0; i < n_classes; ++i)
	{
		parent_prob[i] = static_cast<double>(parent.getCounts(i)) / parent.getNumberOfSamples();
		left_prob[i] = static_cast<double>(left.getCounts(i)) / left.getNumberOfSamples();
		right_prob[i] = static_cast<double>(right.getCounts(i)) / right.getNumberOfSamples();

		if (parent_prob[i] > 0)
		{
			parent_entoropy += -parent_prob[i] * std::log2(parent_prob[i]);
		}
	}

	double left_entoropy = 0;
	double right_entoropy = 0;

	for (int i = 0; i < n_classes; ++i) {
		if (left_prob[i] > 0) left_entoropy += -left_prob[i] * std::log2(left_prob[i]);
		if (right_prob[i] > 0) right_entoropy += -right_prob[i] * std::log2(right_prob[i]);
	}

	double gain = parent_entoropy - static_cast<double>(left.getNumberOfSamples()) / parent.getNumberOfSamples() * left_entoropy
		- static_cast<double>(right.getNumberOfSamples()) / parent.getNumberOfSamples() * right_entoropy;

	return gain;
}

int partitionByResponse(std::vector<int>& indices, int from, int to, std::vector<double>& response, double threshold)
{
	assert(from < to);
	int i = from;
	int j = to - 1;

	while (i <= j)
	{
		if (response[i - from] >= threshold)
		{
			std::swap(indices[i], indices[j]);
			std::swap(response[i - from], response[j - from]);
			--j;
		}
		else ++i;
	}

	return response[i - from] >= threshold ? i : i + 1;
}


template <typename FeatureType>
class DecisionTree
{
public:

    typedef Node::NodePtr NodePtr;
    typedef Node::NodeRawPtr NodeRawPtr;

    DecisionTree(int n_classes_, int n_features, int n_thresholds, int max_d)
        : n_classes(n_classes_),
        n_candidate_feat(n_features),
        n_thres_per_feat(n_thresholds),
        max_depth(max_d),
        n_nodes(0)
    {

    }

    template <typename FeatureContainer,
              typename LabelContainer,
              typename FeatureFactory>
    void train(const FeatureContainer& X,
        const LabelContainer& y,
        std::vector<int>& indices,
        const FeatureFactory& factory,
        const std::vector<double>& class_weights)
    {
        buildTree(X, y, indices, factory, class_weights);
    }

    template <typename D>
    std::vector<double> predictDistribution(const D& x) const
    {
        int index = 0;
        while (!nodes[index]->isLeaf())
        {
            SplitNode<FeatureType>* node = dynamic_cast<SplitNode<FeatureType>*>(nodes[index].get());
            const double response = node->getFeatureResponse(x);
            if (response < node->getThreshold()) index = nodes[index]->getLeftChildIndex();
            else index = nodes[index]->getRightChildIndex();
        }
        return dynamic_cast<LeafNode*>(nodes[index].get())->getDistribution();
    }

private:

    struct NodeBuildInfo
    {
        NodeBuildInfo(int from_, int to_, int parent_index_, bool is_left_, int depth_) :
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

    template <typename FeatureContainer, typename LabelContainer>
    void buildTree(const FeatureContainer& X,
        const LabelContainer& y,
        std::vector<int>& indices,
        const std::function<FeatureType()>& factory,
        const std::vector<double>& class_weights)
    {
        std::queue<NodeBuildInfo> que;
        que.push(NodeBuildInfo(0, indices.size(), 0, false, 0));
        bool is_root = true;

        while (!que.empty())
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

            std::vector<double> response(n_data), best_response(n_data);
            double best_gain = -1;
            double best_thres;
            int best_feature;

            int n_threshold;
            if (n_thres_per_feat == -1 || n_data < n_thres_per_feat)
            {
                n_threshold = n_data - 1;
            }
            else
            {
                n_threshold = n_thres_per_feat - 1;
            }

            std::vector<double> threshold(n_threshold + 1);
            Histogram parent_hist(n_classes, class_weights);
            int prev_label = y[indices[from]];
            bool same_label = true;

            for (int i = from; i < to; ++i)
            {
                int l = y[indices[i]];
                parent_hist.accumulate(l);
                if (l != prev_label) same_label = false;
                prev_label = l;
            }

            if (max_depth <= depth || same_label || n_data <= 5)
            {
                LeafNode* leaf = new LeafNode(node_index, depth, parent_hist);
                nodes.push_back(NodePtr(leaf));
                ++n_nodes;

                if (is_left)
                {
                    nodes[parent_index]->setLeftChildIndex(node_index);
                }
                else
                {
                    nodes[parent_index]->setRightChildIndex(node_index);
                }

                continue;
            }

            const std::vector<FeatureType> candidate_features = generateRandomFeatures(factory, n_candidate_feat);

            for (int i = 0; i < n_candidate_feat; ++i)
            {
                FeatureType f(candidate_features[i]);
                for (int j = from; j < to; ++j)
                {
                    response[j - from] = f(X[indices[j]]);
                }

                generateCandidateThreshold(threshold, response, n_threshold, n_data);
                std::vector<Histogram> partition_statistics(threshold.size(), Histogram(n_classes, class_weights));

                for (int j = from; j < to; ++j)
                {
                    int t = std::upper_bound(threshold.begin(), threshold.begin() + n_threshold, response[j - from]) - threshold.begin();
                    partition_statistics[t].accumulate(y[indices[j]]);
                }

                Histogram left_statistics(n_classes, class_weights), right_statistics(n_classes, class_weights);
                left_statistics.accumulate(partition_statistics[0]);

                for (int t = 1; t < threshold.size(); ++t)
                {
                    right_statistics.accumulate(partition_statistics[t]);
                }

                double gain = computeInfomationGain(parent_hist, left_statistics, right_statistics);
                if (gain > best_gain)
                {
                    best_gain = gain;
                    best_feature = i;
                    best_thres = threshold[0];
                    best_response = response;
                }

                for (int t = 1; t < n_threshold; ++t)
                {
                    left_statistics.accumulate(partition_statistics[t]);
                    right_statistics.decrease(partition_statistics[t]);

                    gain = computeInfomationGain(parent_hist, left_statistics, right_statistics);

                    if (gain > best_gain)
                    {
                        best_gain = gain;
                        best_feature = i;
                        best_thres = threshold[t];
                        best_response = response;
                    }
                }
            }

            if (best_gain <= 0.01)
            {
                LeafNode* leaf = new LeafNode(node_index, depth, parent_hist);
                nodes.push_back(NodePtr(leaf));

                ++n_nodes;
                if (is_left)
                {
                    nodes[parent_index]->setLeftChildIndex(node_index);
                }
                else
                {
                    nodes[parent_index]->setRightChildIndex(node_index);
                }

                continue;
            }

            SplitNode<FeatureType>* split(new SplitNode<FeatureType>(node_index, depth, candidate_features[best_feature], best_thres));
            nodes.push_back(NodePtr(split));
            ++n_nodes;

            if (is_root)
            {
                is_root = false;
            }
            else
            {
                if (is_left)
                {
                    nodes[parent_index]->setLeftChildIndex(node_index);
                }
                else
                {
                    nodes[parent_index]->setRightChildIndex(node_index);
                }
            }

            int thres_index = partitionByResponse(indices, from, to, best_response, best_thres);

            que.push(NodeBuildInfo(from, thres_index, node_index, true, depth));
            que.push(NodeBuildInfo(thres_index, to, node_index, false, depth));

        }
    }

    const int n_classes;
    const int n_candidate_feat;
    const int n_thres_per_feat;
    const int max_depth;
    int n_nodes;
    std::vector<NodePtr> nodes;
};
