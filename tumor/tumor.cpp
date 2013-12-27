#include <memory>
#include <cstdlib>
#include <iostream>
#include <functional>
#include <map>
#include <tuple>
#include <omp.h>
#include <boost/chrono/chrono.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/program_options.hpp>
#include <boost/timer.hpp>
#include <itkImage.h>
#include <itkImageFileWriter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkImageFileReader.h>
#include <forest/RandomForest.h>
#include <util/eigen.h>
#include <gmm/GMM.h>

template <typename T>
using Volume = itk::Image<T, 3>;

template <typename T>
using VolumePtr = typename Volume<T>::Pointer;

template <typename T>
using Index = typename Volume<T>::IndexType;

template <typename T>
using VolumeVector = std::vector<VolumePtr<T>>;

typedef std::tuple<VolumeVector<short>, VolumeVector<double>, VolumePtr<unsigned char>> Instance;
typedef std::vector<Instance> DataSet;

struct DataInstance
{
    DataInstance(const VolumeVector<short>& vols, const VolumePtr<short>& g, int x_, int y_, int z_) :
    volumes(vols),
    gt(g),
    x(x_),
    y(y_),
    z(z_)
    {
    }

    const VolumeVector<short>& volumes;
    const VolumePtr<short>& gt;
    const int x, y, z;
};

using namespace std;
using namespace boost::chrono;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

fs::path findMHA(const fs::path& dir_path) {
    const fs::directory_iterator end;
    const auto it = find_if(fs::directory_iterator(dir_path), end,
        [](const fs::directory_entry& e)
    {
        return boost::algorithm::ends_with(e.path().string(), "mha");
    }
    );
    return it->path();
}

template <typename T>
VolumePtr<T> createVolume(int width, int height, int depth)
{
    itk::Index<3> start;
    start.Fill(0);
    itk::Size<3> dim;
    dim[0] = width;
    dim[1] = height;
    dim[2] = depth;

    VolumePtr<T> vol = Volume<T>::New();
    typename Volume<T>::RegionType region(start, dim);
    vol->SetRegions(region);
    vol->Allocate();
    
    return vol;
}

template <typename T>
std::tuple<int, int, int> getVolumeDimension(const VolumePtr<T>& vol)
{
    const auto size = vol->GetLargestPossibleRegion().GetSize();
    const int width = size[0];
    const int height = size[1];
    const int depth = size[2];

    return std::make_tuple(width, height, depth);
}

void addInstance(const fs::path& instance_path, std::vector<VolumeVector<short>>& mrs, VolumeVector<unsigned char>& masks, VolumeVector<short>& gts, std::vector<std::vector<double>>& gmm_data)
{
    const fs::path flair_path(instance_path / fs::path("VSD.Brain.XX.O.MR_Flair"));
    const fs::path t1_path(instance_path / fs::path("VSD.Brain.XX.O.MR_T1"));
    const fs::path t1c_path(instance_path / fs::path("VSD.Brain.XX.O.MR_T1c"));
    const fs::path t2_path(instance_path / fs::path("VSD.Brain.XX.O.MR_T2"));
    const fs::path gt_path(instance_path / fs::path("VSD.Brain_3more.XX.XX.OT"));

    vector<fs::path> volume_paths =
    {
        flair_path,
        t1_path,
        t1c_path,
        t2_path
    };

    typedef itk::ImageFileReader<Volume<short>> ImageReader;
    ImageReader::Pointer gt_reader = ImageReader::New();
    gt_reader->SetFileName(findMHA(gt_path).string());
    gt_reader->Update();
    VolumePtr<short> gt = gt_reader->GetOutput();
    gts.push_back(gt);
    VolumeVector<short> volumes;

    int width, height, depth;
    std::tie(width, height, depth) = getVolumeDimension<short>(gt);

    for (const fs::path& p : volume_paths)
    {
        ImageReader::Pointer reader = ImageReader::New();
        const fs::path mha_path = findMHA(p);
        reader->SetFileName(mha_path.string());
        reader->Update();
        volumes.push_back(reader->GetOutput());
    }

    mrs.push_back(volumes);

    typedef itk::BinaryThresholdImageFilter<Volume<short>, Volume<unsigned char>> ThresholdFilter;
    ThresholdFilter::Pointer thres = ThresholdFilter::New();
    thres->SetInput(volumes[0]);
    thres->SetLowerThreshold(1);
    thres->SetInsideValue(1);
    thres->SetOutsideValue(0);
    thres->Update();

    VolumePtr<unsigned char> mask = thres->GetOutput();
    masks.push_back(mask);
    //const char* filename = "mask.mha";
    //typedef itk::ImageFileWriter<MaskVolume> Writer;
    //Writer::Pointer writer = Writer::New();
    //writer->SetInput(mask);
    //writer->SetFileName(filename);
    //writer->Update();

    //const unsigned char* mask_ptr = mask->GetBufferPointer();
    //const short* gt_ptr = gt->GetBufferPointer();
    //std::vector<const short*> volume_ptrs;
    //for (VolumePtr ptr : volumes) volume_ptrs.push_back(ptr->GetBufferPointer());

    std::array<double, 5> rate = { 0.01, 0.6, 0.1, 1.0, 0.3 };
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                Index<short> index;
                index[0] = x;
                index[1] = y;
                index[2] = z;
                if (mask->GetPixel(index) == 0) continue;
                const short label = gt->GetPixel(index);
                assert(label <= 4);
                const double p = (double)rand() / RAND_MAX;
                if (p > rate[label]) continue;
                for (int i = 0; i < 4; ++i)
                {
                    const double v = static_cast<double>(volumes[i]->GetPixel(index));
                    gmm_data[label].push_back(v);
                }
            }
        }
    }

}

int main(int argc, char *argv[])
{
    po::options_description opt("option");
    opt.add_options()
        ("data_dir", po::value<string>()->default_value("HG"))
        ("patch_size", po::value<int>()->default_value(5))
        ("subsample", po::value<int>()->default_value(3))
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opt), vm);
    po::notify(vm);

    const fs::path data_dir(vm["data_dir"].as<string>());
    fs::directory_iterator dir_iter(data_dir);

    std::vector<VolumeVector<short>> mr_data;
    std::vector<VolumePtr<short>> gts;
    std::vector<VolumePtr<unsigned char>> masks;
    std::vector<std::vector<double>> gmm_data(5);

    int c = 0;
    const int n_training_instances = 12;
    for (; dir_iter != fs::directory_iterator(); ++dir_iter, ++c)
    {
        if (c == n_training_instances) break;
        const fs::path instance(dir_iter->path());
        addInstance(instance, mr_data, masks, gts, gmm_data);
        if(c == n_training_instances);
    }


    const int n_classes = 5;
    const int n_components = 5;
    const int n_mr_channels = 4;
    std::vector<GMM<double>> gmms(n_classes, GMM<double>(n_components));

    const int n_threads = 8;
    omp_set_num_threads(n_threads);
    const auto start = high_resolution_clock::now();

#pragma omp parallel for
    for (int i = 0; i < n_classes; ++i)
    {
        //    std::cout << "Training " << i << " th gmm.\n";
        const int rows = gmm_data[i].size() / n_mr_channels;
        Matrix<double> X = MatrixMapper<double>(&gmm_data[i][0], n_mr_channels, rows).transpose();
        gmms[i].train(X);
    }

    const auto end = high_resolution_clock::now();
    const duration<double> t = end - start;
    std::cout << t.count() << std::endl;
    // return 0;

    const auto start2 = high_resolution_clock::now();
    DataSet data(mr_data.size());

#pragma omp parallel for
    for (int i = 0; i < mr_data.size(); ++i)
    {
        int width, height, depth;
        std::tie(width, height, depth) = getVolumeDimension<short>(mr_data[i][0]);

        std::vector<VolumePtr<double>> gmm_volumes(n_classes);
        for (int j = 0; j < n_classes; ++j)
        {
            gmm_volumes[j] = createVolume<double>(width, height, depth);
        }

        for (int z = 0; z < depth; ++z)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const int ix = x + y * width + z * width * height;
                    //          std::cout << ix << std::endl;
                    Index<double> index;
                    index[0] = x;
                    index[1] = y;
                    index[2] = z;

                    bool inside = true;
                    if (masks[i]->GetPixel(index) == 0) inside = false;

                    Vector<double> X(n_mr_channels);
                    for (int c = 0; c < n_mr_channels; ++c)
                    {
                        X(c) = static_cast<double>(mr_data[i][c]->GetPixel(index));
                    }

                    for (int j = 0; j < n_classes; ++j)
                    {
                        double v = 0;
                        if (inside) v = gmms[j].evaluate(X);
                        gmm_volumes[j]->SetPixel(index, v);
                    }
                }
            }
        }

        data[i] = std::make_tuple(mr_data[i], gmm_volumes, masks[i]);
    }

    const auto end2 = high_resolution_clock::now();
    const duration<double> t2 = end2 - start2;
    std::cout << t2.count() << std::endl;

    return 0;
}


