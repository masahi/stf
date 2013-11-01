#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <cassert>

class Histogram
{
public:

    Histogram(int n_classes)
        : n_bins(n_classes),
          n_samples(0),
          bins(n_classes,0)
    {

    }

    Histogram(const Histogram& other)
        : n_bins(other.getNumberOfBins()),
          n_samples(other.getNumberOfSamples()),
          bins(other.getBins())
    {
    }

    Histogram& operator=(const Histogram& other)
    {
        n_samples = other.getNumberOfSamples();
        n_bins = other.getNumberOfBins();
        bins = other.getBins();
        return *this;
    }

    int getNumberOfSamples() const
    {
        return n_samples;
    }

    int getNumberOfBins() const
    {
        return n_bins;
    }

    std::vector<int> getBins() const
    {
        return bins;
    }

    int getCounts(int i) const
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
        ++bins[label];
        ++n_samples;
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
        --bins[label];
        --n_samples;
    }

    void clear()
    {
        n_samples = 0;
        std::fill(bins.begin(), bins.end(), 0);
    }

private:
    int n_bins;
    int n_samples;
    std::vector<int> bins;
};

#endif // HISTOGRAM_H
