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

    Node(const Histogram& h)
        : feature(FeatureType::getRandom()),
          threshold(0),
          hist(h),
          dist(hist.getNumberOfBins(), 0),
          is_leaf(true)
    {
        const int n_samples = hist.getNumberOfSamples();

        std::vector<int> bins = hist.getBins();
        std::transform(bins.begin(), bins.end(), dist.begin(), [](int count){ return static_cast<double>(count)/n_samples;});
    }

    Node(const FeatureType& f, double t,const Histogram& h)
        : feature(f),
          threshold(t),
          hist(h),
          is_leaf(false)
    {
    }

    Node& operator=(const Node& other)
    {
        feature(other.getFeature());
        return *this;
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

    void setLeftChild(NodePtr& child)
    {
        left = std::move(child);
    }

    void setRightChild(NodePtr& child)
    {
        right = std::move(child);
    }

    NodeRawPtr getLeftChild() const { return left.get();}
    NodeRawPtr getRightChild() const { return right.get();}
    double getThreshold() const { return threshold;}
    FeatureType getFeature() const { return feature;}
    std::vector<double> getDistribution() const { return dist;}
    bool isLeaf() const { return is_leaf;}

private:
    FeatureType feature;
    double threshold;
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
    void train(const std::vector<D>& X, const std::vector<int>& y, std::vector<int>& indices)
    {
        root = buildTree(X, y, indices, 0, y.size());
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
    NodePtr buildTree(const std::vector<D>& X, const std::vector<int>& y, std::vector<int>& indices, int from, int to)
    {
        const int n_data = to - from;
    //    std::cout << from << "," << to << "," << n_data << std::endl;
        std::vector<double> response(n_data);

        double best_gain = -1;
        FeatureType best_feature;
        double best_thres;

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

            std::cout << "same labels!\n";
            NodePtr leaf(new Node<FeatureType>(parent_hist));
            return leaf;

            }
        }

 if(n_data <=5 )
        {
            std::cout << "min sample reached.!!\n";
            NodePtr leaf(new Node<FeatureType>(parent_hist));
            return leaf;
        }


        std::vector<double> threshold(n_thres_per_feat+1,std::numeric_limits<double>::max());
        int n_threshold;
        for(int i = 0; i < n_candidate_feat; ++i)
        {
            FeatureType f = FeatureType::getRandom();
            for(int j = from; j < to; ++j)
            {
                response[j-from] = f(X[indices[j]]);
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
                std::cout << "Response values all the same." << std::endl;
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
            std::cout << "Gain zero!!\n";
            NodePtr leaf(new Node<FeatureType>(parent_hist));
            return leaf;
        }

        for(int i = from; i < to; ++i)
        {
            response[i-from] = best_feature(X[indices[i]]);
        }

     //   std::cout << best_gain << std::endl;
        NodePtr parent(new Node<FeatureType>(best_feature, best_gain, parent_hist));

        int index = 0;
        int thres_index = std::partition(indices.begin()+from, indices.begin()+to, [&](int dummy){index++; return response[index-1] < best_thres;}) - indices.begin();//-from;
     //   int thres_index = partitionByResponse(indices,from, to, response, best_thres);

        //recurese on left and right child
        NodePtr l_child = buildTree(X, y, indices, from, thres_index);
        NodePtr r_child = buildTree(X, y, indices, thres_index, to);
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
