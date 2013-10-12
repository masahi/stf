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

    IdentityFeature(int index_)
        : index(index_)
    {
    }

    template <typename T>
    double operator()(const std::vector<T>& v) const
    {
        return v[index];
    }

private:
    const int index;
};

#endif // FEATURE_H
