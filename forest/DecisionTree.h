#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include <Histogram.h>
#include <util.h>

template <typename FeatureType>
class Node
{
public:

    typedef std::unique_ptr<Node<FeatureType>> NodePtr;

    Node(const Histogram& h)
        : hist(h),
          is_leaf(true)
    {

    }

    Node(const FeatureType& f, double t,const Histogram& h)
        : feature(f),
          threshold(t),
          hist(h),
          is_leaf(false)
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

    void setLeftChild(NodePtr child)
    {
        left = child;
    }

    void setRightChild(NodePtr child)
    {
        right = child;
    }


private:
    const FeatureType& feature;
    const double threshold;
    Histogram hist;
    bool is_leaf;
    NodePtr left;
    NodePtr right;
};

template <typename FeatureType>
class DecisionTree
{
public:

    typedef typename Node<FeatureType>::NodePtr NodePtr;

    DecisionTree(int n_classes_)
        : n_classes(n_classes_),
          n_candidate_feat(100),
          n_thres_per_feat(100)
    {
    }

    ~DecisionTree()
    {

    }

    template <typename D>
    void train(const std::vector<D>& X, const std::vector<int>& y, std::vector<int>& indices)
    {
        root = buildTree(X, y, indices, 0, y.size());
    }


private:

    template <typename D>
    NodePtr buildTree(const std::vector<D>& X, const std::vector<int>& y, std::vector<int>& indices, int from, int to)
    {
        const int n_data = to - from;
        std::vector<double> response(n_data);

        double best_gain = -1;
        FeatureType best_feature;
        double best_thres;

        Histogram parent_hist;
        for(int i = from; i < to; ++i)
        {
            parent_hist.accumulate(y[indices[from]]);
        }
        for(int i = 0; i < n_candidate_feat; ++i)
        {
            FeatureType f = FeatureType::getRandom();
            for(int j = from; j < to; ++j)
            {
                response[j] = f(X[indices[j]]);
            }

            std::vector<double> threshold(n_thres_per_feat+1);
            int n_threshold;
            if(n_data > n_thres_per_feat)
            {
                for(int j = 0; j < n_thres_per_feat+1; ++j)
                {
                    threshold[j] = response[randInt(from, to)];
                    n_threshold = n_thres_per_feat;
                }
            }
            else
            {
                std::copy(&response[from], &response[to], threshold.begin());
                n_threshold = n_data;
            }

            std::sort(threshold.begin(), threshold.end());

            if(threshold[0] == threshold[n_threshold-1])
            {
                std::cout << "Response values all the same." << std::endl;
                continue;
            }

            for(int j = 0; j < n_threshold; ++j)
            {
                threshold[j] = threshold[j] + rand() * (threshold[j+1] - threshold[j]);
            }

            std::vector<Histogram> partition_statistics(n_threshold);
            for(int j = from; j < to; ++j)
            {
                int t = std::upper_bound(threshold.begin(), threshold.end(), response[j]) - threshold.begin();
                partition_statistics[t].accumulate(y[indices[j]]);
            }

            Histogram left_statistics, right_statistics;
            for(int t = 0; t < n_threshold; ++t)
            {
                left_statistics.clear();
                right_statistics.clear();

                for(int p = 0; p < n_threshold+1; ++p)
                {
                    if(p <= t)
                    {
                        left_statistics.accumulate(partition_statistics[p]);
                    }
                    else
                    {
                        right_statistics.accumulate(partition_statistics[p]);
                    }
                }


            double gain = computeInfomationGain(parent_hist, left_statistics, right_statistics);
            if(gain > best_gain)
            {
                best_gain = gain;
                best_feature = f;
                best_thres = t;
            }
            }

        }

        if(best_gain <= 0.01)
        {
            std::cout << "Gain zero!!\n";
            NodePtr leaf(new Node<FeatureType>(parent_hist));
            return leaf;
        }

        for(int i = from; i < to; ++i)
        {
            response[i] = best_feature(X[indices[i]]);
        }

        NodePtr parent(new Node<FeatureType>(best_feature, best_gain, parent_hist));
        int thres_index = partition(indices, from, to, response, best_thres);
        //recurese on left and right child
        NodePtr l_child = buildTree(X, y, from, thres_index);
        NodePtr r_child = buildTree(X, y, thres_index, to);
        parent->setLeftChild(l_child);
        parent->setRightChild(r_child);
        return parent;
    }


    const int n_classes;
    const int n_nodes;
    const int n_candidate_feat;
    const int n_thres_per_feat;
    NodePtr root;
};
