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

    enum class ResponseType
    {
        Unary,
        Add,
        Sub,
        AbsSub
    };

    PatchFeature(int patch_size_, ResponseType type_):
        patch_size(patch_size_),
        type(type_)
    {

    }

    double operator()(const cv::Mat& patch)
    {
        return 0;

    }


private:

    const int patch_size;
    const ResponseType type;
};

IdentityFeature* createFeature(int dim)
{
    return new IdentityFeature(randInt(0, dim));
}

#endif // FEATURE_H
