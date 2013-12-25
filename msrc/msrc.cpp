#include <memory>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <functional>
#include <map>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/timer.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <GCoptimization.h>
#include <forest/RandomForest.h>
#include <util.h>

using namespace cv;
using namespace std;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

int main(int argc, char *argv[])
{
    po::options_description opt("option");
    opt.add_options()
        ("img_dir", po::value<string>())
        ("gt_dir", po::value<string>())
        ("train_split", po::value<string>()->default_value("split/Train.txt"))
        ("test_split", po::value<string>()->default_value("split/Test.txt"))
        ("patch_size", po::value<int>()->default_value(5))
        ("subsample", po::value<int>()->default_value(3))
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opt), vm);
    po::notify(vm);

    ConfigMap msrc_config
    {
        { Vec3b(0, 0, 0), -1, "void" },
        { Vec3b(0, 0, 128), 0, "building" },
        { Vec3b(0, 128, 0), 1, "grass" },
        { Vec3b(0, 128, 128), 2, "tree" },
        { Vec3b(128, 0, 0), 3, "cow" },
        { Vec3b(128, 128, 0), 4, "sheep" },
        { Vec3b(128, 128, 128), 5, "sky" },
        { Vec3b(0, 0, 192), 6, "airplane" },
        { Vec3b(0, 128, 64), 7, "water" },
        { Vec3b(0, 128, 192), 8, "face" },
        { Vec3b(128, 0, 64), 9, "car" },
        { Vec3b(128, 0, 192), 10, "bicycle" },
        { Vec3b(128, 128, 64), 11, "flower" },
        { Vec3b(128, 128, 192), 12, "sign" },
        { Vec3b(0, 64, 0), 13, "bird" },
        { Vec3b(0, 64, 128), 14, "book" },
        { Vec3b(0, 192, 0), 15, "chair" },
        { Vec3b(128, 64, 128), 16, "road" },
        { Vec3b(128, 192, 0), 17, "cat" },
        { Vec3b(128, 192, 128), 18, "dog" },
        { Vec3b(0, 64, 64), 19, "body" },
        { Vec3b(0, 64, 192), 20, "boot" },
        { Vec3b(0, 0, 64), 21, "mountain" },
        { Vec3b(128, 0, 128), 22, "sheep" }
    };

    const BgrMap& bgr_map = msrc_config.get<bgr>();
    const LabelMap& label_map = msrc_config.get<label>();

    ifstream train_split(vm["train_split"].as<string>());
    ifstream test_split(vm["test_split"].as<string>());
    const fs::path img_dir(vm["img_dir"].as<string>());
    const fs::path gt_dir(vm["gt_dir"].as<string>());
    fs::directory_iterator img_file(img_dir);
    fs::directory_iterator gt_file(gt_dir);
    const int patch_size = vm["patch_size"].as<int>();

    std::vector<string> img_paths;
    std::vector<string> gt_paths;

    string line;
    while(std::getline(train_split, line))
    {
        const string img_path = replaceString((img_dir / fs::path(line)).string(), ".bmp\r", ".bmp");
        const string gt_path = replaceString((gt_dir / fs::path(line)).string(), ".bmp\r", "_GT.bmp");
        img_paths.push_back(img_path);
        gt_paths.push_back(gt_path);
    }

    const int n_training_imgs = img_paths.size();

    while(std::getline(test_split,line))
    {
        const string img_path = replaceString((img_dir / fs::path(line)).string(), ".bmp\r", ".bmp");
        const string gt_path = replaceString((gt_dir / fs::path(line)).string(), ".bmp\r", "_GT.bmp");
        img_paths.push_back(img_path);
        gt_paths.push_back(gt_path);
    }


    vector<int> all_labels;
    vector<Mat> all_patches;
    const int subsample = vm["subsample"].as<int>();

    for (int i = 0; i < n_training_imgs; ++i)
    {
        const Mat img = imread(img_paths[i]);
        const Mat gt = imread(gt_paths[i]);

        vector<int> labels;
        vector<Mat> patches;

        tie(patches, labels) = extractPatches(img, gt, bgr_map, patch_size, subsample);

        append(all_patches, patches);
        append(all_labels, labels);
    }

    std::vector<double> counts(24, 0);
    for (size_t i = 0; i < all_labels.size(); i++)
    {
        counts[all_labels[i]] += 1;
    }
    std::cout << all_patches.size() << std::endl;
    std::cout << bgr_map.size() << std::endl;

    for(int i = 0; i < 24; ++i) std::cout << counts[i] << std::endl;

    const std::function<PatchFeature ()> factory = std::bind(createPatchFeature, patch_size);

    const int n_classes = msrc_config.size();
    const int n_trees = 8;
    const int n_features = 400;//static_cast<int>(std::sqrt(feature_dim));
    const int n_thres = 5;
    const int max_depth = 10;
    RandomForest<PatchFeature> forest(n_classes, n_trees, n_features, n_thres, max_depth);

    std::vector<double> class_weight(24,0);
    for(int i = 0; i < 24; ++i) class_weight[i] = 1.0 / counts[i];
    forest.train(all_patches, all_labels, factory, class_weight);

    std::cout << "Done training\n";

    int n_tests = 0;
    std::vector<int> n_tests_per_class(n_classes, 0);
    int n_corrects = 0;
    std::vector<int> n_corrects_per_class(n_classes, 0);
    int n_corrects_gco = 0;
    std::vector<int> n_corrects_per_class_gco(n_classes, 0);

    const double w = 64;
    std::vector<double> smooth_cost(n_classes * n_classes, 64);
    for (size_t i = 0; i < n_classes; i++)
    {
        smooth_cost[i + i * n_classes] = 0;
    }

    const fs::path output_dir("output");
    const fs::path output_dir_gco("output_gco");

    for (size_t i = n_training_imgs; i < img_paths.size(); ++i)
    {
        std::cout << i << std::endl;
        const Mat img = imread(img_paths[i]);
        const Mat gt = imread(gt_paths[i]);

        const int rows = img.rows;
        const int cols = img.cols;

        vector<int> labels;
        vector<Mat> patches;

        std::vector<double> unary_cost;

        tie(patches, labels) = extractPatches(img, gt, bgr_map, patch_size, 1, true);

        Mat label_image(rows, cols, CV_8UC3);
        Mat label_image_gco(rows, cols, CV_8UC3);

        for (size_t p = 0; p < patches.size(); ++p)
        {
            const int prediction = forest.predict(patches[p]);
            const int label = labels[p];

            const std::vector<double> dist = forest.predictDistribution(patches[p]);
            append(unary_cost, dist);

            n_tests += 1;
            n_corrects += (prediction == label ? 1 : 0);
            n_tests_per_class[label] += 1;
            n_corrects_per_class[label] += (prediction == label ? 1 : 0);
            label_image.at<Vec3b>(p / cols, p % cols) = label_map.find(prediction-1)->bgr;
        }

//        const boost::scoped_ptr<GCoptimization> gco(new GCoptimizationGridGraph(cols, rows, n_classes));
//        gco->setDataCost(&unary_cost[0]);
//        gco->setSmoothCost(&smooth_cost[0]);
//        gco->expansion();

//        for (size_t r = 0; r < rows; ++r)
//        {
//            for (size_t c = 0; c < cols; ++c)
//            {
//                const int prediction = gco->whatLabel(c + r * cols);
//                const int label = labels[c + r * n_classes];
//                n_corrects_gco += (prediction == label ? 1 : 0);
//                n_corrects_per_class_gco[label] += (prediction == label ? 1 : 0);

//                label_image_gco.at<Vec3b>(r, c) = label_map.find(prediction-1)->bgr;
//            }
//        }

        const fs::path save_path = fs::current_path() / output_dir / fs::path(img_paths[i]);
//        const fs::path save_path_gco = fs::current_path() / output_dir_gco / fs::path(img_paths[i]);

        std::cout << save_path.string() << std::endl;
        imwrite(save_path.string(), label_image);
 //       imwrite(save_path_gco.string(), label_image_gco);

    }

    std::cout << "******************* Unary Only ***********************\n";
    std::cout << "Overall accuracy: " << static_cast<double>(n_corrects) / n_tests << std::endl;
    std::cout << "Individual accuracy:" << std::endl;

    for (size_t i = 0; i < n_classes; i++)
    {
        std::cout << label_map.find(i-1)->name << ": " << static_cast<double>(n_corrects_per_class[i]) / n_tests_per_class[i] << std::endl;
    }

//    std::cout << "******************* GCO ***********************\n";
//    std::cout << "Overall accuracy: " << static_cast<double>(n_corrects_gco) / n_tests << std::endl;
//    std::cout << "Individual accuracy:" << std::endl;

//    for (size_t i = 0; i < n_classes; i++)
//    {
//        std::cout << label_map.find(i-1)->name << ": " << static_cast<double>(n_corrects_per_class_gco[i]) / n_tests_per_class[i] << std::endl;
//    }
    return 0;
}


