#ifndef FEATURE_H
#define FEATURE_H

class PrecomputedFeature
{
public:

    PrecomputedFeature(const int index): featureIndex(index)
    {
    }

    template <typename T>
    double operator()(const std::vector<T>& v) const
    {
        return v[featureIndex];
    }

    template <typename Vector>
    double operator()(const Vector& v) const
    {
        return v(featureIndex);
    }



private:
    const int featureIndex;
};

#endif // FEATURE_H
