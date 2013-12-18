#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <cassert>

class Histogram
{
public:

    Histogram(int n_classes, const std::vector<double>& weights_)
        : n_bins(n_classes),
          n_samples(0),
          bins(n_classes,0),
          weights(weights_)
    {

    }

    double getNumberOfSamples() const
    {
        return n_samples;
    }

    int getNumberOfBins() const
    {
        return n_bins;
    }

    std::vector<double> getBins() const
    {
        return bins;
    }

    double getCounts(int i) const
    {
        assert(i < n_bins);
        return bins[i];
    }

    void accumulate(const Histogram& other)
    {
        assert(bins.size() == other.getNumberOfBins());
        n_samples += other.getNumberOfSamples();
        for(int i = 0; i < n_bins; ++i)
        {
            bins[i] += other.getCounts(i);
        }
    }

    void accumulate(int label)
    {
        assert(label < n_bins);
        assert(label >= 0);
        bins[label] += weights[label];
        n_samples += weights[label];
    }


    void decrease(const Histogram& other)
    {
        assert(bins.size() == other.getNumberOfBins());
        n_samples -= other.getNumberOfSamples();
        for(int i = 0; i < n_bins; ++i)
        {
            bins[i] -= other.getCounts(i);
        }
    }

    void decrease(int label)
    {
        assert(label < n_bins);
        bins[label] -= weights[label];
        n_samples -= weights[label];
    }

    void clear()
    {
        n_samples = 0;
        std::fill(bins.begin(), bins.end(), 0);
    }

private:
    const int n_bins;
    double n_samples;
    std::vector<double> bins;
    std::vector<double> weights;
};

#endif // HISTOGRAM_H
