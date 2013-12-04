#include <RandomForest.h>
#include <feature.h>
#include <memory>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <boost/timer.hpp>
#include <util.h>
#include <map>
#include <GCoptimization.h>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/scoped_ptr.hpp>
#include <boost/multi_index_container.hpp>
#include <opencv2/highgui/highgui.hpp>

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
		("patch_size", po::value<int>()->default_value(5))
		("subsample", po::value<int>()->default_value(3))
		;

	po::variables_map vm;
	po::store(po::parse_command_line(argc, argv, opt), vm);
	po::notify(vm);

	map<Vec3b, int> bgr2label
	{
		{ Vec3b(0, 0, 0), -1 },
		{ Vec3b(0, 0, 128), 0 },
		{ Vec3b(0, 128, 0), 1 },
		{ Vec3b(0, 128, 128), 2 },
		{ Vec3b(128, 0, 0), 3 },
		{ Vec3b(128, 128, 0), 4 },
		{ Vec3b(128, 128, 128), 5 },
		{ Vec3b(0, 0, 192), 6 },
		{ Vec3b(0, 128, 64), 7 },
		{ Vec3b(0, 128, 192), 8 },
		{ Vec3b(128, 0, 64), 9 },
		{ Vec3b(128, 0, 192), 10 },
		{ Vec3b(128, 128, 64), 11 },
		{ Vec3b(128, 128, 192), 12 },
		{ Vec3b(0, 64, 0), 13 },
		{ Vec3b(0, 64, 128), 14 },
		{ Vec3b(0, 192, 0), 15 },
		{ Vec3b(128, 64, 128), 16 },
		{ Vec3b(128, 192, 0), 17 },
		{ Vec3b(128, 192, 128), 18 },
		{ Vec3b(0, 64, 64), 19 },
		{ Vec3b(0, 64, 192), 20 },
		{ Vec3b(0, 0, 64), 21 },
		{ Vec3b(128, 0, 128), 22 }
	};

	map<int, string> label2name
	{
	};

	map<int, Vec3b> label2bgr
	{
	};

	const fs::path img_dir(vm["img_dir"].as<string>());
	const fs::path gt_dir(vm["gt_dir"].as<string>());
	fs::directory_iterator img_file(img_dir);
	fs::directory_iterator gt_file(gt_dir);
	const int patch_size = vm["patch_size"].as<int>();

	std::vector<string> img_paths;
	std::vector<string> gt_paths;

	for (; img_file != fs::directory_iterator(); ++img_file, ++gt_file)
	{
		const string img_path = img_file->path().string();
		const string gt_path = gt_file->path().string();

		img_paths.push_back(img_path);
		gt_paths.push_back(gt_path);
	}

	std::sort(img_paths.begin(), img_paths.end());
	std::sort(gt_paths.begin(), gt_paths.end());

	vector<int> all_labels;
	vector<Mat> all_patches;
	const int subsample = vm["subsample"].as<int>();

	for (int i = 0; i < img_paths.size(); i += 2)
	{
		std::cout << i << std::endl;
		const Mat img = imread(img_paths[i]);
		const Mat gt = imread(gt_paths[i]);


		vector<int> labels;
		vector<Mat> patches;

		tie(patches, labels) = extractPatches(img, gt, bgr2label, patch_size, subsample);

		append(all_patches, patches);
		append(all_labels, labels);
	}

	std::cout << all_patches.size() << std::endl;

	return 0;

	const std::function<PatchFeature* ()> factory = std::bind(createPatchFeature, patch_size);
	const int n_trees = 1;
	const int n_classes = bgr2label.size();

	RandomForest<PatchFeature> forest(n_trees, n_classes);
	forest.train(all_patches, all_labels, factory);

	int n_tests = 0;
	std::vector<int> n_tests_per_class(bgr2label.size() - 1, 0);
	int n_corrects = 0;
	std::vector<int> n_corrects_per_class(bgr2label.size() - 1, 0);
	int n_corrects_gco = 0;
	std::vector<int> n_corrects_per_class_gco(bgr2label.size() - 1, 0);

	std::vector<double> smooth_cost(n_classes * n_classes, 0);
	const double w = 64;
	for (size_t i = 0; i < n_classes; i++)
	{
		smooth_cost[i + i * n_classes] = w;
	}

	for (size_t i = 1; i < img_paths.size(); i += 2)
	{
		std::cout << i << std::endl;
		const Mat img = imread(img_paths[i]);
		const Mat gt = imread(gt_paths[i]);

		const int rows = img.rows;
		const int cols = img.cols;

		vector<int> labels;
		vector<Mat> patches;

		std::vector<double> unary_cost;

		tie(patches, labels) = extractPatches(img, gt, bgr2label, patch_size, 1, true);

		Mat label_image(rows, cols, CV_8UC3);
		Mat label_image_gco(rows, cols, CV_8UC3);

		for (size_t i = 0; i < patches.size(); i++)
		{
			const int prediction = forest.predict(patches[i]);
			const int label = labels[i];

			const std::vector<double> dist = forest.predictDistribution(patches[i]);
			append(unary_cost, dist);

			n_tests += 1;
			n_corrects += (prediction == label ? 1 : 0);
			n_tests_per_class[label] += 1;
			n_corrects_per_class[label] += (prediction == label ? 1 : 0);
			label_image.at<Vec3b>(i / cols,i % cols) = label2bgr[prediction];
		}

		boost::scoped_ptr<GCoptimization> gco(new GCoptimizationGridGraph(cols, rows, n_classes));
		gco->setDataCost(&unary_cost[0]);
		gco->setSmoothCost(&smooth_cost[0]);
		gco->expansion();

		for (size_t i = 0; i < rows; i++)
		{
			for (size_t j = 0; j < cols; j++)
			{
				const int prediction = gco->whatLabel(j + i * n_classes);
				const int label = labels[j + i * n_classes];
				n_corrects_gco += (prediction == label ? 1 : 0);
				n_corrects_per_class_gco[label] += (prediction == label ? 1 : 0);

				label_image_gco.at<Vec3b>(i,j) = label2bgr[prediction];
			}
		}

		const string save_path("");
		const string save_path_gco("");
		imwrite(save_path, label_image);
		imwrite(save_path_gco, label_image_gco);
	}

	std::cout << "******************* Unary Only ***********************\n";
	std::cout << "Overall accuracy: " << static_cast<double>(n_corrects) / n_tests << std::endl;
	std::cout << "Individual accuracy:" << std::endl;

	for (size_t i = 0; i < bgr2label.size(); i++)
	{
		std::cout << label2name[i] << ": " << static_cast<double>(n_corrects_per_class[i]) / n_tests_per_class[i] << std::endl;
	}

	std::cout << "******************* GCO ***********************\n";
	std::cout << "Overall accuracy: " << static_cast<double>(n_corrects_gco) / n_tests << std::endl;
	std::cout << "Individual accuracy:" << std::endl;

	for (size_t i = 0; i < bgr2label.size(); i++)
	{
		std::cout << label2name[i] << ": " << static_cast<double>(n_corrects_per_class_gco[i]) / n_tests_per_class[i] << std::endl;
	}
	return 0;
}

