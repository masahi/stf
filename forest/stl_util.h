#ifndef STL_UTIL_H
#define STL_UTIL_H

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <opencv2/core/core.hpp>
#include <eigen_util.h>

namespace std
{
    bool operator<(const cv::Vec3b& lhs, const cv::Vec3b& rhs)
    {
        int index = 0;

        while (index < 3)
        {
            if (lhs[index] < rhs[index]) return true;
            else if (lhs[index] > rhs[index]) return false;
            ++index;
        }

        return false;

    }

    template<>
    struct less<cv::Vec3b>
    {
        bool operator()(const cv::Vec3b& lhs, const cv::Vec3b& rhs) const
        {
            int index = 0;

            while (index < 3)
            {
                if (lhs[index] < rhs[index]) return true;
                else if (lhs[index] > rhs[index]) return false;
                ++index;
            }

            return false;

        }
    };
}

struct Config
{
    const cv::Vec3b bgr;
    const int label;
    const std::string name;
};

struct bgr{};
struct label{};
struct name{};

typedef boost::multi_index::multi_index_container<
    Config,
    boost::multi_index::indexed_by<
    boost::multi_index::ordered_unique<boost::multi_index::tag<bgr>, boost::multi_index::member<Config, const cv::Vec3b, &Config::bgr>>,
    boost::multi_index::ordered_unique<boost::multi_index::tag<label>, boost::multi_index::member<Config, const int, &Config::label>>
    >
> ConfigMap;

typedef ConfigMap::index<bgr>::type BgrMap;
typedef ConfigMap::index<label>::type LabelMap;

int randInt(int a, int b)
{
    std::random_device seed;
    std::default_random_engine engine(seed());
    std::uniform_int_distribution<> dist(a, b - 1);
    return dist(engine);
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
#endif // STL_UTIL_H
