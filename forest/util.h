#ifndef UTIL_H
#define UTIL_H

#include <random>
#include <algorithm>

#include <iostream>
#include <string>
#include <cassert>

#include <set>

#include <map>
#include <sstream>
#include <vector>



#include <stl_util.h>
#include <eigen_util.h>
#include <Histogram.h>



std::vector<int> randomSamples(int m, int n)
{
	std::vector<int> indices(n);

	for (int i = 0; i < n; ++i) {
		indices[i] = randInt(0, m);
	}
	return indices;
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




#endif // UTIL_H
