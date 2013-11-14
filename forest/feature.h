#ifndef FEATURE_H
#define FEATURE_H

#include <util.h>
#include <opencv2/core/core.hpp>

class IdentityFeature
{
public:
   IdentityFeature(int i) : index(i){}

   template <typename T>
   double operator()(const std::vector<T>& v) const { return v[index];}

   template <typename T>
   double operator()(const Vector<T>& v) const { return v(index);}

private:
   int index;
};

class PatchFeature
{
public:

    enum Type
    {
        Unary,
        Add,
        Sub,
        AbsSub
    };

    PatchFeature(){}

    template
    double operator()(const cv::Mat& patch)
    {

    }


private:

    const int patch_size;
    const Type type
};

IdentityFeature* createFeature(int dim)
{
    return new IdentityFeature(randInt(0, dim));
}

#endif // FEATURE_H
