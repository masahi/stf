#ifndef FEATURE_H
#define FEATURE_H

#include <util.h>

class IdentityFeature
{
public:

    static int FEATURE_DIM;

    static IdentityFeature getRandom()
    {
        return IdentityFeature(randInt(0, FEATURE_DIM));
    }

    IdentityFeature()
        : index(-1)
    {
    }

    IdentityFeature(int index_)
        : index(index_)
    {
    }

    IdentityFeature& operator=(const IdentityFeature& other)
    {
        index = other.getIndex();
        return *this;
    }

    template <typename T>
    double operator()(const std::vector<T>& v) const
    {
        assert(index >= 0);
        assert(index < FEATURE_DIM);
        return v[index];
    }

    int getIndex() const
    {
        return index;
    }

private:
    int index;
};

#endif // FEATURE_H
