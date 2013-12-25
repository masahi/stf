#include <vector>
#include <string>
#include <tuple>
#include <sstream>
#include <fstream>
#include <boost/algrithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <Eigen/Core>

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
