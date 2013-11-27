#include <RandomForest.h>
#include <feature.h>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <boost/timer.hpp>
#include <util.h>
#include <map>
//#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
//namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description opt("option");
    opt.add_options()
            ("img_dir", po::value<string>())
            ("gt_dir", po::value<string>())
            ("patch_size", po::value<int>()->default_value(5))
            ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc,argv,opt), vm);
    po::notify(vm);

    map<Vec3b, int> rgb2label
    {
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

//    const fs::path img_dir(vm["img_dir"].as<string>());
//    const fs::path gt_dir(vm["gt_dir"].as<string>());
//    fs::directory_iterator img_file(img_dir);
//    fs::directory_iterator gt_file(gt_dir);
//    const int patch_size = vm["patch_size"].as<int>();

//    for (; img_file != fs::directory_iterator(); ++img_file, ++gt_file)
//    {
////        const Mat img = imread(img_file->string());
////        const Mat gt = imread(gt_file->string());

////        vector<int> labels;
////        vector<Mat> patches;

////        tie(patches, labels) = extractPatches(img,gt,rgb2label,patch_size);
//    }

    return 0;
}

