#ifndef FEATURE_H
#define FEATURE_H

#include <util.h>
#include <vector>
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

    enum ResponseType
    {
        Unary,
        Add,
        Sub,
        AbsSub
    };

    PatchFeature(int patch_size_, ResponseType type_, const cv::Point2d& pos1_, const cv::Point2d& pos2_, int channel_):
        patch_size(patch_size_),
        type(type_),
        pos1(pos1_),
        pos2(pos2_),
        channel(channel_)
    {
    }

    double operator()(const cv::Mat& patch) const
    {
        const uchar v1 = patch.at<cv::Vec3b>(pos1)[channel];
        const uchar v2 = patch.at<cv::Vec3b>(pos2)[channel];

        if(type == Unary) return v1;
        else if(type == Add) return v1 + v2;
        else if(type == Sub) return v1 - v2;
        else return std::abs(v1 - v2);
    }


private:

    int patch_size;
    ResponseType type;
    cv::Point2d pos1,pos2;
    int channel;
};

IdentityFeature createFeature(int dim)
{
    return  IdentityFeature(randInt(0, dim));
}

PatchFeature createPatchFeature(int patch_size)
{
    const int r = randInt(0,4);
    PatchFeature::ResponseType type;

    if(r == 0) type = PatchFeature::Unary;
    else if(r == 1) type = PatchFeature::Add;
    else if(r == 2) type = PatchFeature::Sub;
    else type = PatchFeature::AbsSub;

    const cv::Point2d pos1(randInt(0, patch_size), randInt(0, patch_size));
    const cv::Point2d pos2(randInt(0, patch_size), randInt(0, patch_size));

    const int channel = randInt(0, 3);

    return PatchFeature(patch_size,type, pos1, pos2, channel);
}


#endif // FEATURE_H
