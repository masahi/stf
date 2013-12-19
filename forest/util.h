#ifndef UTIL_H
#define UTIL_H

#include <random>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <vector>
#include <set>
#include <tuple>
#include <map>
#include <sstream>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stl_util.h>
#include <eigen_util.h>
#include <Histogram.h>

template <typename FeatureType>
std::vector<FeatureType> generateRandomFeatures(const std::function<FeatureType()>& factory, int n)
{
    std::vector<FeatureType> features(n, factory());
    std::generate(features.begin(), features.end(), factory);
    return features;
}

std::vector<double> generateCandidateThreshold(const std::vector<double>& response, int n_threshold, int n_data)
{
    std::vector<double> threshold(n_threshold + 1);
    if (n_data == n_threshold + 1)
    {
        std::copy(response.begin(), response.end(), threshold.begin());
    }
    else
    {
        for (int j = 0; j < threshold.size(); ++j)
        {
            threshold[j] = response[randInt(0, n_data)];
        }
    }

    std::sort(threshold.begin(), threshold.end());

    for (int j = 0; j < n_threshold; ++j)
    {
        threshold[j] = threshold[j] + static_cast<double>(rand()) / RAND_MAX * (threshold[j + 1] - threshold[j]);
    }

    return threshold;
}
double computeInfomationGain(const Histogram& parent, const Histogram& left, const Histogram& right)
{
	const int n_classes = parent.getNumberOfBins();
	std::vector<double> parent_prob(n_classes, 0);
	std::vector<double> left_prob(n_classes, 0);
	std::vector<double> right_prob(n_classes, 0);
	double parent_entoropy = 0;
	for (int i = 0; i < n_classes; ++i)
	{
		parent_prob[i] = static_cast<double>(parent.getCounts(i)) / parent.getNumberOfSamples();
		left_prob[i] = static_cast<double>(left.getCounts(i)) / left.getNumberOfSamples();
		right_prob[i] = static_cast<double>(right.getCounts(i)) / right.getNumberOfSamples();

		if (parent_prob[i] > 0)
		{
			parent_entoropy += -parent_prob[i] * std::log2(parent_prob[i]);
		}
	}

	double left_entoropy = 0;
	double right_entoropy = 0;

	for (int i = 0; i < n_classes; ++i) {
		if (left_prob[i] > 0) left_entoropy += -left_prob[i] * std::log2(left_prob[i]);
		if (right_prob[i] > 0) right_entoropy += -right_prob[i] * std::log2(right_prob[i]);
	}

	double gain = parent_entoropy - static_cast<double>(left.getNumberOfSamples()) / parent.getNumberOfSamples() * left_entoropy
		- static_cast<double>(right.getNumberOfSamples()) / parent.getNumberOfSamples() * right_entoropy;

	return gain;
}


std::vector<int> randomSamples(int m, int n)
{
	std::vector<int> indices(n);

	for (int i = 0; i < n; ++i) {
		indices[i] = randInt(0, m);
	}
	return indices;
}

template <typename T>
std::tuple<std::vector<std::vector<T>>, std::vector<int> > readLibsvm(const std::string& file, int dim)
{
	std::ifstream ifs(file.c_str());
	std::string buf;
	std::vector<std::string> line;
	std::vector<std::vector<T>> features;
	std::vector<int> label;
	std::vector<T> feature(dim, 0);

	while (std::getline(ifs, buf))
	{
		line.clear();
		boost::split(line, buf, boost::is_any_of(" \n\t"));
		int c = boost::lexical_cast<int>(line[0]);
		label.push_back(c);
		for (int i = 1; i < line.size(); ++i)
		{
			std::istringstream is(line[i]);
			int f;
			T v;
			char colon;

			is >> f >> colon >> v;
			assert(f <= dim);

			feature[f - 1] = v;
		}

		features.push_back(feature);
	}

	auto iter = std::find(label.begin(), label.end(), 0);
	if (iter == label.end())
	{
		std::transform(label.begin(), label.end(), label.begin(), [](int i){return i - 1; });
	}
	std::replace(label.begin(), label.end(), -2, 1);

	return std::make_tuple(features, label);
}

template <typename T>
std::tuple<Matrix<T>, Eigen::VectorXi> readLibsvmEigen(const std::string& file, int dim)
{
	std::vector<std::vector<T>> X;
	std::vector<int> y;

	std::tie(X, y) = readLibsvm<double>(file, dim);
	Matrix<T> m(X.size(), dim);
	for (int i = 0; i < y.size(); ++i)
	{
		Vector<T> vec = Eigen::Map<Vector<T>>(&X[i][0], dim);
		m.row(i) = vec.transpose();
	}

	Eigen::VectorXi l = Eigen::Map<Eigen::VectorXi>(&y[0], y.size());

	return std::make_tuple(m, l);
}


template<typename T>
int countUnique(const std::vector<T>& vec)
{
	std::set<T> s(vec.begin(), vec.end());
	return s.size();

}

int partitionByResponse(std::vector<int>& indices, int from, int to, std::vector<double>& response, double threshold)
{
	assert(from < to);
	int i = from;
	int j = to - 1;

	while (i <= j)
	{
		if (response[i - from] >= threshold)
		{
			std::swap(indices[i], indices[j]);
			std::swap(response[i - from], response[j - from]);
			--j;
		}
		else ++i;
	}

	return response[i - from] >= threshold ? i : i + 1;
}

std::tuple<std::vector<cv::Mat>, std::vector<int> > extractPatches(const cv::Mat& img, const cv::Mat& gt, const BgrMap& bgr_map, int patch_size, int subsample = 1, bool WITH_BORDER = false, bool TRANSFORM = 1)
{
	std::vector<cv::Mat> patches;
	std::vector<int> labels;

	const int rad = patch_size / 2;
	const int rows = img.rows;
	const int cols = img.cols;

	cv::Mat padded;
	cv::copyMakeBorder(img, padded, rad, rad, rad, rad, cv::BORDER_REFLECT);

	int r_begin, r_end, c_begin, c_end;
	if (WITH_BORDER)
	{
		r_begin = rad;
		r_end = rows + rad;
		c_begin = rad;
		c_end = cols + rad;
	}
	else
	{
		r_begin = patch_size;
		r_end = rows;
		c_begin = patch_size;
		c_end = cols;
	}

	int count = 0;
	for (int r = r_begin; r < r_end; ++r)
	{
		for (int c = c_begin; c < c_end; ++c, ++count)
		{
			if (count % subsample == 0)
			{
				cv::Rect roi(c - rad, r - rad, patch_size, patch_size);
				const cv::Vec3b bgr = gt.at<cv::Vec3b>(r - rad, c - rad);

				if (bgr_map.find(bgr) == bgr_map.end())
				{
					std::cout << (int)bgr[0] << "," << (int)bgr[1] << "," << (int)bgr[2] << std::endl;
					std::cout << r - rad << "," << c - rad << std::endl;
				}
				else
				{
//					std::cout << (int)bgr[0] << "," << (int)bgr[1] << "," << (int)bgr[2] << ",";
//					std::cout << r - rad << "," << c - rad << std::endl;
//					std::cout << bgr_map.find(bgr)->label << std::endl;
                    labels.push_back(bgr_map.find(bgr)->label+1);
					patches.push_back(padded(roi));
				}

				if (TRANSFORM)
				{

				}
			}
		}
	}

	return std::make_tuple(patches, labels);
}



#endif // UTIL_H
