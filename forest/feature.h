#ifndef FEATURE_H
#define FEATURE_H

class IdentityFeature
{
public:

    IdentityFeature(int index_)
        :index(index_)
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
