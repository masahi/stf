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
#include <itkImageFileReader.h>
#include <forest/RandomForest.h>
#include <util/eigen.h>
#include <gmm/GMM.h>

typedef itk::Image<short, 3> Volume;
typedef Volume::Pointer VolumePtr;
typedef itk::Image<double, 3> DoubleVolume;
typedef DoubleVolume::Pointer DoubleVolumePtr;
typedef Volume::IndexType Index;
typedef itk::ImageFileReader<Volume> ImageReader;
typedef std::vector<std::vector<VolumePtr>> MRDataSet;
typedef std::vector<std::tuple<std::vector<VolumePtr>, std::vector<DoubleVolumePtr>>> DataSet;

struct DataInstance
{
    DataInstance(const std::vector<VolumePtr>& vols, const VolumePtr& g, int x_, int y_, int z_) :
    volumes(vols),
    gt(g),
    x(x_),
    y(y_),
    z(z_)
    {
    }

    const std::vector<VolumePtr>& volumes;
    const VolumePtr& gt;
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

void addInstance(const fs::path& instance_path, MRDataSet& data, std::vector<VolumePtr>& gts, std::vector<std::vector<double>>& gmm_data)
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

    ImageReader::Pointer gt_reader = ImageReader::New();
    gt_reader->SetFileName(findMHA(gt_path).string());
    gt_reader->Update();
    VolumePtr gt = gt_reader->GetOutput();
    gts.push_back(gt);
    vector<VolumePtr> volumes;

    const auto size = gt->GetLargestPossibleRegion().GetSize();
    const int width = size[0];
    const int height = size[1]; 
    const int depth = size[2];

    for (const fs::path& p : volume_paths)
    {
        ImageReader::Pointer reader = ImageReader::New();
        const fs::path mha_path = findMHA(p);
        reader->SetFileName(mha_path.string());
        reader->Update();
        volumes.push_back(reader->GetOutput());
    }

    data.push_back(volumes);

    std::array<double,5> rate = {0.01, 0.6, 0.1, 1.0, 0.3};
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                Index index;
                index[0] = x;
                index[1] = y;
                index[2] = z;
                const short label = gt->GetPixel(index);
                assert(label <= 4);
                if (label > 4) continue;
                double p = (double)rand()/RAND_MAX;
                if(p > rate[label]) continue;
                std::vector<double> values(4);
                bool inside = true;
                for (int i = 0; i < 4; ++i)
                {
                    const double v = static_cast<double>(volumes[i]->GetPixel(index));
                    if(label == 0 && v == 0) inside = false;
                    else values[i] = v;
                }
                if(!inside) continue;
                else append(gmm_data[label], values);
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
    MRDataSet mr_data;
    std::vector<VolumePtr> gts;
    std::vector<std::vector<double>> gmm_data(5);

    int c = 0;
    const int n_training_instances = 12;
    for (; dir_iter != fs::directory_iterator(); ++dir_iter, ++c)
    {
        if(c == n_training_instances) break;
        const fs::path instance(dir_iter->path());
        addInstance(instance, mr_data, gts, gmm_data);
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
        const auto size = mr_data[i][0]->GetLargestPossibleRegion().GetSize();
        const int width = size[0];
        const int height = size[1];
        const int depth = size[2];

        std::vector<DoubleVolumePtr> gmm_volumes;
        itk::Index<3> start;
        start.Fill(0);
        itk::Size<3> dim;
        dim[0] = width;
        dim[1] = height;
        dim[2] = depth;
        DoubleVolume::RegionType region(start, dim);
        
        for (int j = 0; j < n_classes; ++j)
        {
            gmm_volumes.push_back(DoubleVolume::New());
            gmm_volumes[j]->SetRegions(region);
            gmm_volumes[j]->Allocate();
        }
        
        for (int z = 0; z < depth; ++z)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    const int ix = x + y * width + z * width * height;
          //          std::cout << ix << std::endl;
                    Index index;
                    index[0] = x;
                    index[1] = y;
                    index[2] = z;

                    Vector<double> X(n_mr_channels);
                    for (int c = 0; c < n_mr_channels; ++c)
                    {
                        X(c) = static_cast<double>(mr_data[i][c]->GetPixel(index));
                    }

                    for (int j = 0; j < n_classes; ++j)
                    {
                        double v = 0;
                        if(X(0) != 0) v = gmms[j].evaluate(X);
                        gmm_volumes[j]->SetPixel(index, v);
                    }
                }
            }
        }

        data[i] = std::make_tuple(mr_data[i], gmm_volumes);
    }

    const auto end2 = high_resolution_clock::now();
    const duration<double> t2 = end2 - start2;
    std::cout << t2.count() << std::endl;

    return 0;
}


