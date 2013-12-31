#ifndef GENERAL_H
#define GENERAL_H

#include <random>
#include <algorithm>
#include <string>
#include <cassert>
#include <vector>
#include <set>
#include <tuple>
#include <map>
#include <util/eigen.h>
#include <chrono>

/*
  TODO: Needs more work here
*/

typedef std::chrono::milliseconds milliseconds;
typedef std::chrono::seconds seconds;
typedef std::chrono::minutes minutes;

template <typename T, typename Func>
double timeit(Func f)
{
    const auto start = std::chrono::system_clock::now();
    f();
    const auto end = std::chrono::system_clock::now();
    const auto t = end - start;
    return std::chrono::duration_cast<T>(t).count();
}

std::string replaceString(const std::string& src, const std::string& od, const std::string& nw)
{
    std::string ret(src);
    std::string::size_type pos = 0;
    while(pos = ret.find(od, pos), pos != std::string::npos) {
        ret.replace(pos, od.length(), nw);
        pos += nw.length();
    }

    return ret;
}

int randInt(int a, int b)
{
    std::random_device seed;
    std::default_random_engine engine(seed());
    std::uniform_int_distribution<> dist(a, b - 1);
    return dist(engine);
}

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

template <typename T>
int argmax(const std::vector<T>& v)
{
    return std::max_element(v.begin(), v.end()) - v.begin();
}

template <typename T>
int argmin(const std::vector<T>& v)
{
    return std::min_element(v.begin(), v.end()) - v.begin();
}

template <typename T>
void append(std::vector<T>& a, const std::vector<T>& b)
{
    a.insert(a.end(), b.begin(), b.end());
}

template <typename T>
std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b)
{
    const size_t n = a.size();
    assert(b.size() == n);

    VectorMapper<T>(&a[0], n) += ConstVectorMapper<T>(&b[0], n);
    return a;
}

template <typename T1, typename T2>
std::vector<T2> operator*(T1 coeff, const std::vector<T2>& v)
{
    const size_t n = v.size();
    std::vector<T2> multiplied(n);
    VectorMapper<T2>(&multiplied[0], n) = coeff * ConstVectorMapper<T2>(&v[0], n);
    return multiplied;
}

template <typename T>
std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b)
{
    const size_t n = a.size();
    assert(b.size() == n);

    std::vector<T> sum(n);
    VectorMapper<T>(&sum[0], n) = ConstVectorMapper<T>(&a[0], n) + ConstVectorMapper<T>(&b[0], n);
    return sum;
}

#endif // GENERAL_H
