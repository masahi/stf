#include <vector>
#include <tuple>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/ordered_index.hpp>

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

