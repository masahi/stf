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
#include <itkImageFileWriter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkImageFileReader.h>
#include <forest/RandomForest.h>
#include <util/eigen.h>
#include <gmm/GMM.h>
#include <util.h>
#include <SpatialFeature.h>

using namespace std;
using namespace boost::chrono;
namespace fs = boost::filesystem;
namespace po = boost::program_options;

typedef std::tuple<VolumeVector<short>, VolumeVector<double>, VolumePtr<unsigned char>> Instance;
typedef std::vector<Instance> DataSet;

struct DataInstance
{
    DataInstance(const Instance& ins, int x_, int y_, int z_) :
    instance(ins),
    x(x_),
    y(y_),
    z(z_)
    {
    }

    const Instance& instance;
    const int x, y, z;
};

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

int addInstance(const fs::path& instance_path, std::vector<VolumeVector<short>>& mrs, VolumeVector<unsigned char>& masks, VolumeVector<short>& gts, std::vector<std::vector<double>>& gmm_data, std::vector<int>& counts)
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
    int n_voxels = 0;
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
                ++n_voxels;
                const short label = gt->GetPixel(index);
                assert(label <= 4);
                ++counts[label];
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

    return n_voxels;
}

int addTrainingInstance(std::vector<DataInstance>& training_data, std::vector<int>& labels, const Instance& instance, const VolumePtr<short>& gt)
{
    const VolumePtr<unsigned char>& mask = std::get<2>(instance);

    int width, height, depth;
    std::tie(width, height, depth) = getVolumeDimension<unsigned char>(mask);

    std::array<double, 5> rate = { 0.05, 1.0, 1.0, 1.0, 1.0 };
    int n_voxels = 0;
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
                training_data.push_back(DataInstance(instance, x, y, z));
                labels.push_back(label);
                ++n_voxels;
            }
        }
    }

    return n_voxels;
}

int main(int argc, char *argv[])
{
    po::options_description opt("option");
    opt.add_options()
        ("data_dir", po::value<string>()->default_value("HG"))
        ("max_box_size", po::value<int>()->default_value(11))
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opt), vm);
    po::notify(vm);

    const fs::path data_dir(vm["data_dir"].as<string>());
    fs::directory_iterator dir_iter(data_dir);

    const int n_classes = 5;
    std::vector<VolumeVector<short>> mr_data;
    std::vector<VolumePtr<short>> gts;
    std::vector<VolumePtr<unsigned char>> masks;
    std::vector<std::vector<double>> gmm_data(5);
    std::vector<int> counts(n_classes, 0);
    int total_voxel = 0;

    int c = 0;
    const int n_training_instances = 15;
    for (; dir_iter != fs::directory_iterator(); ++dir_iter, ++c)
    {
        if (c == n_training_instances) break;
        const fs::path instance(dir_iter->path());
        total_voxel += addInstance(instance, mr_data, masks, gts, gmm_data, counts);
    }

    std::vector<double> prior(n_classes);
    for (int i = 0; i < n_classes; ++i)
    {
        prior[i] = static_cast<double>(counts[i]) / total_voxel;
    }

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
        const Matrix<double> X = ConstMatrixMapper<double>(&gmm_data[i][0], n_mr_channels, rows).transpose();
        gmms[i].train(X);
    }

    const auto end = high_resolution_clock::now();
    const duration<double> t = end - start;
    std::cout << t.count() << std::endl;

    const auto start2 = high_resolution_clock::now();
    DataSet data(mr_data.size());

#pragma omp parallel for
    for (int i = 0; i < mr_data.size(); ++i)
    {
        int width, height, depth;
        std::tie(width, height, depth) = getVolumeDimension<short>(mr_data[i][0]);
        std::vector<double> prob(n_classes);

        auto spacing = gts[i]->GetSpacing();
        auto origin = gts[i]->GetOrigin();

        std::cout << origin << std::endl;
        std::vector<VolumePtr<double>> gmm_volumes(n_classes);
        for (int j = 0; j < n_classes; ++j)
        {
            gmm_volumes[j] = createVolume<double>(width, height, depth, spacing, origin);
        }

        for (int z = 0; z < depth; ++z)
        {
            for (int y = 0; y < height; ++y)
            {
                for (int x = 0; x < width; ++x)
                {
                    Index<double> index;
                    index[0] = x;
                    index[1] = y;
                    index[2] = z;

                    if (masks[i]->GetPixel(index) == 0) continue;

                    Vector<double> X(n_mr_channels);
                    for (int c = 0; c < n_mr_channels; ++c)
                    {
                        X(c) = static_cast<double>(mr_data[i][c]->GetPixel(index));
                    }

                    double normalizer = 0;
                    for (int j = 0; j < n_classes; ++j)
                    {
                        const double p = gmms[j].evaluate(X) * prior[j];
                        prob[j] = p;
                        normalizer += p;
                    }

                    for (int j = 0; j < n_classes; ++j)
                    {
                        gmm_volumes[j]->SetPixel(index, prob[j] / normalizer);
                    }

                }
            }
        }

        data[i] = std::make_tuple(mr_data[i], gmm_volumes, masks[i]);
    }

    const auto end2 = high_resolution_clock::now();
    const duration<double> t2 = end2 - start2;
    std::cout << t2.count() << std::endl;

    std::vector<DataInstance> training_data;
    std::vector<int> labels;
    int n_data = 0;
    for (int i = 0; i < data.size(); ++i)
    {
        n_data += addTrainingInstance(training_data, labels, data[i], gts[i]);
    }

    const int n_trees = 8;
    const int n_features = 100;
    const int n_thres = 10;
    const int max_depth = 10;
    RandomForest<SpatialFeature> forest(n_classes, n_trees, n_features, n_thres, max_depth);

    const int max_box_size = vm["max_box_size"].as<int>();
    std::function<SpatialFeature()> factory = std::bind(createSpatialFeature, max_box_size);

    std::vector<int> class_counts(n_classes, 0);
    for (int label : labels) ++class_counts[label];
    std::vector<double> weights(n_classes);
    for (int i = 0; i < n_classes; ++i)
    {
        weights[i] = 1.0 / class_counts[i];
    }

    for(int c: class_counts) std::cout << c << std::endl;
    for(double w: weights) std::cout << w << std::endl;
    std::cout << n_data << std::endl;

    const double sample_rate = 0.25;
    forest.train(training_data, labels, factory, weights, sample_rate);

    std::cout << "Done Training.\n";

    /* TESTING */
    std::vector<int> n_tests_per_class(n_classes, 0);
    std::vector<int> n_corrects_per_class(n_classes, 0);

    for (; dir_iter != fs::directory_iterator(); ++dir_iter)
    {
        const fs::path instance_path(dir_iter->path());

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

        int width, height, depth;
        std::tie(width, height, depth) = getVolumeDimension<short>(gt);
        auto spacing = gt->GetSpacing();
        auto origin = gt->GetOrigin();

        std::vector<double> prob(n_classes);
        std::vector<VolumePtr<double>> gmm_volumes(n_classes);
        for (int j = 0; j < n_classes; ++j)
        {
            gmm_volumes[j] = createVolume<double>(width, height, depth, spacing, origin);
        }

        VolumeVector<short> volumes;
        for (const fs::path& p : volume_paths)
        {
            ImageReader::Pointer reader = ImageReader::New();
            const fs::path mha_path = findMHA(p);
            reader->SetFileName(mha_path.string());
            reader->Update();
            volumes.push_back(reader->GetOutput());
        }

        typedef itk::BinaryThresholdImageFilter<Volume<short>, Volume<unsigned char>> ThresholdFilter;
        ThresholdFilter::Pointer thres = ThresholdFilter::New();
        thres->SetInput(volumes[0]);
        thres->SetLowerThreshold(1);
        thres->SetInsideValue(1);
        thres->SetOutsideValue(0);
        thres->Update();

        VolumePtr<unsigned char> mask = thres->GetOutput();

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

                    Vector<double> X(n_mr_channels);
                    for (int c = 0; c < n_mr_channels; ++c)
                    {
                        X(c) = static_cast<double>(volumes[c]->GetPixel(index));
                    }

                    double normalizer = 0;
                    for (int j = 0; j < n_classes; ++j)
                    {
                        const double p = gmms[j].evaluate(X) * prior[j];
                        prob[j] = p;
                        normalizer += p;
                    }

                    for (int j = 0; j < n_classes; ++j)
                    {
                        gmm_volumes[j]->SetPixel(index, prob[j] / normalizer);
                    }
                }
            }
        }

        const Instance ins = std::make_tuple(volumes, gmm_volumes, mask);
        VolumePtr<unsigned char> prediction = createVolume<unsigned char>(width, height, depth, spacing, origin);

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
                    if (mask->GetPixel(index) == 0)
                    {
                        prediction->SetPixel(index, 0);
                        continue;
                    }

                    DataInstance data(ins, x, y, z);
                    const int pred = forest.predict(data);
                    prediction->SetPixel(index, pred);

                    const int true_label = gt->GetPixel(index);
                    ++n_tests_per_class[true_label];
                    n_corrects_per_class[true_label] += (pred == true_label ? 1 : 0);
                }
            }
        }

        const string filename = "seg.mha";
        const fs::path save_path = fs::absolute(instance_path) / fs::path(filename);
        typedef itk::ImageFileWriter<Volume<unsigned char>> Writer;
        Writer::Pointer writer = Writer::New();
        writer->SetInput(prediction);
        writer->SetFileName(save_path.string());
        writer->Update();
    }

    std::cout << "******** Accuracy **********\n";
    for (int i = 0; i < n_classes; ++i)
    {
        std::cout << static_cast<double>(n_corrects_per_class[i]) / n_tests_per_class[i];
    }
    return 0;
}


