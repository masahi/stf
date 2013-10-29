#ifndef UTIL_H
#define UTIL_H

#include <random>
#include <algorithm>
#include <Histogram.h>

int randInt(int a, int b)
{
    std::random_device seed;
    std::default_random_engine engine(seed());
    std::uniform_int_distribution<> dist(a, b-1);
    return dist(engine);
}

double computeInfomationGain(const Histogram& parent, const Histogram& left, const Histogram& right)
{
    return 0;
}

int partition(std::vector<int>& indices, int from, int to, const std::vector<double>& response, double threshold)
{
    int i = from;
    int j = to-1;

    while(i <= j)
    {
        if(response[i] >= threshold)
        {
            std::swap(indices[i], indices[j]);
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
