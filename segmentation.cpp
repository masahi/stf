#include <RandomForest.h>
#include <feature.h>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <boost/timer.hpp>
#include <util.h>
#include <map>

using namespace cv;
using namespace std;

int main(int argc, char *argv[])
{
    const string image_file(argv[1]);
    const string gt_file(argv[2]);

    Mat img = imread(image_file);
    Mat gt = imread(gt_file);

    vector<int> labels;
    vector<Mat> patches;
    map<Vec3b, int> rgb2label{
        {Vec3b(0,0,128), 0},
        {Vec3b(0,128,0), 1},
        {Vec3b(0,128,128),2},
        {Vec3b(128,0,0), 3},
        {Vec3b(128,128,128), 5},
        {Vec3b(0,0,192), 6},
        {Vec3b(0,128,64), 7},
        {Vec3b(0,128,192), 8},
        {Vec3b(128,0,64), 9},
        {Vec3b(128,0,192),10},
        {Vec3b(128,128,64), 11},
        {Vec3b(128,128,192), 12},
        {Vec3b(0,64,0), 13},
        {Vec3b(0,64,128), 14},
        {Vec3b(0,192,0), 15},
        {Vec3b(128,64,128), 16},
        {Vec3b(128,192,0), 17},
        {Vec3b(128,192,128), 18},
        {Vec3b(0,64,64), 19},
        {Vec3b(0,64,192), 20}
    };

    tie(patches, labels) = extractPatches(img,gt,rgb2label);
    return 0;
}

