#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <vector>
#include <cassert>

class Histogram
{
public:


    int getNumberOfSamples() const
    {
        return nSamples;
    }

    int getNumberOfBins() const
    {
        return nBins;
    }

    int getCounts(int i) const
    {
        assert(i < nBins);
        return bins[i];
    }

    void accumulate(const Histogram& other)
    {
        assert(nBins == other.getNumberOfBins());
        nSamples += other.getNumberOfSamples();
        for(int i = 0; i < nBins; ++i)
        {
            bins[i] += other.getCounts(i);
        }
    }

    void accumulate(int label)
    {
        assert(label < nBins);
        ++bins[label];
        ++nSamples;
    }

    void clear()
    {
        nSamples = 0;
        bins.clear();
    }

private:
    const int nBins;
    int nSamples;
    std::vector<int> bins;
};

#endif // HISTOGRAM_H
