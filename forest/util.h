#ifndef UTIL_H
#define UTIL_H

#include <random>
#include <algorithm>
#include <Histogram.h>
#include <vector>

int randInt(int a, int b)
{
    std::random_device seed;
    std::default_random_engine engine(seed());
    std::uniform_int_distribution<> dist(a, b-1);
    return dist(engine);
}

double computeInfomationGain(const Histogram& parent, const Histogram& left, const Histogram& right)
{
    const int n_classes = parent.getNumberOfBins();
    std::vector<double> parent_prob(n_classes,0);
    std::vector<double> left_prob(n_classes,0);
    std::vector<double> right_prob(n_classes,0);
    double parent_entoropy = 0;
    for(int i = 0; i < n_classes; ++i)
    {
        parent_prob[i] = static_cast<double>(parent.getCounts(i)) / parent.getNumberOfSamples();
        left_prob[i] = static_cast<double>(left.getCounts(i)) / left.getNumberOfSamples();
        right_prob[i] = static_cast<double>(right.getCounts(i)) / right.getNumberOfSamples();

        if(parent_prob[i] > 0)
        {
            parent_entoropy += -parent_prob[i] * std::log2(parent_prob[i]);
        }
    }

    double left_entoropy = 0;
    double right_entoropy = 0;

    for (int i = 0; i < n_classes; ++i) {
        if(left_prob[i] > 0) left_entoropy += -left_prob[i] * std::log2(left_prob[i]);
        if(left_prob[i] > 0) right_entoropy += -right_prob[i] * std::log2(right_prob[i]);
    }

    double gain = parent_entoropy - static_cast<double>(left.getNumberOfSamples()) / parent.getNumberOfSamples() * left_entoropy
            - static_cast<double>(right.getNumberOfSamples()) / parent.getNumberOfSamples() * right_entoropy;

    return gain;
}

int partition(std::vector<int>& indices, int from, int to, std::vector<double>& response, double threshold)
{
    assert(from < to);
    int i = from;
    int j = to-1;

    while(i <= j)
    {
        if(response[i] >= threshold)
        {
            std::swap(indices[i], indices[j]);
            std::swap(response[i], response[j]);
            --j;
        }
        else ++i;
    }

    return response[i] >= threshold ? i : i+1;
}

std::vector<int> randomSamples(int m, int n)
{
    std::vector<int> samples(n);
    return samples;
}

#endif // UTIL_H
