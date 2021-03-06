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
#include <itkMinimumMaximumImageCalculator.h>
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

typedef std::tuple<VolumeVector<short>, VolumeVector<double>, VolumePtr<unsigned char>, VolumeVector<int>, VolumeVector<double>, VolumePtr<unsigned int>> Instance;
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

    const vector<fs::path> volume_paths =
    {
        flair_path,
        t1_path,
        t1c_path,
        t2_path
    };

    const VolumePtr<short> gt = readVolume<short>(findMHA(gt_path).string());
    gts.push_back(gt);
    VolumeVector<short> volumes;

    int width, height, depth;
    std::tie(width, height, depth) = getVolumeDimension<short>(gt);

    for (const fs::path& p : volume_paths)
    {
        volumes.push_back(readVolume<short>(findMHA(p).string()));
    }

    mrs.push_back(volumes);

    const short lower_thres = 1;
    const VolumePtr<unsigned char> mask = createMask<short>(volumes[0], lower_thres);
    masks.push_back(mask);

    const std::array<double, 5> rate = { 0.01, 0.6, 0.1, 1.0, 0.3 };
    int n_voxels = 0;
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                itk::Index<3> index;
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

    const std::array<double, 5> rate = { 0.05, 1.0, 1.0, 1.0, 1.0 };
    int n_voxels = 0;
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                itk::Index<3> index;
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
        ("max_box_size", po::value<int>()->default_value(31))
        ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, opt), vm);
    po::notify(vm);

    const fs::path data_dir(vm["data_dir"].as<string>());
    fs::directory_iterator dir_iter(data_dir);

    const int n_classes = 5;
    const int n_mr_channels = 4;
    const int n_components = 5;
    const int n_trees = 8;
    const int n_features = 100;
    const int n_thres = 10;
    const int max_depth = 10;
    std::vector<double> prior(n_classes);

    std::vector<GMM<double>> gmms(n_classes, GMM<double>(n_components));
    RandomForest<SpatialFeature> forest(n_classes, n_trees, n_features, n_thres, max_depth);

    {
        std::vector<VolumeVector<short>> mr_data;
        VolumeVector<short> gts;
        VolumeVector<unsigned char> masks;
        std::vector<std::vector<double>> gmm_data(n_classes);
        std::vector<int> counts(n_classes, 0);
        int total_voxel = 0;

        int c = 0;
        const int n_training_instances = 5;
        for (; dir_iter != fs::directory_iterator(); ++dir_iter, ++c)
        {
            if (c == n_training_instances) break;
            const fs::path instance(dir_iter->path());
            std::cout << instance.string() << std::endl;
            total_voxel += addInstance(instance, mr_data, masks, gts, gmm_data, counts);
        }

        for (int i = 0; i < n_classes; ++i)
        {
            prior[i] = static_cast<double>(counts[i]) / total_voxel;
        }

        const int n_threads = 8;
        omp_set_num_threads(n_threads);

#pragma omp parallel for
        for (int i = 0; i < n_classes; ++i)
        {
            const int rows = gmm_data[i].size() / n_mr_channels;
            const Matrix<double> X = ConstMatrixMapper<double>(&gmm_data[i][0], n_mr_channels, rows).transpose();
            gmms[i].train(X);
        }

        DataSet data(mr_data.size());

#pragma omp parallel for
        for (int i = 0; i < mr_data.size(); ++i)
        {
            int width, height, depth;
            std::tie(width, height, depth) = getVolumeDimension<short>(mr_data[i][0]);
            std::vector<double> prob(n_classes);

            const auto spacing = gts[i]->GetSpacing();
            const auto origin = gts[i]->GetOrigin();

            std::vector<VolumePtr<double>> gmm_volumes(n_classes);
            for (int j = 0; j < n_classes; ++j)
            {
                gmm_volumes[j] = createVolume<double>(width, height, depth, spacing, origin);
                gmm_volumes[j]->FillBuffer(0);
            }

            VolumeVector<int> integral_mr_volumes(n_mr_channels);
            for (int j = 0; j < n_mr_channels; ++j)
            {
                integral_mr_volumes[j] = createVolume<int>(width, height, depth, spacing, origin);
                integral_mr_volumes[j]->FillBuffer(0);
            }

            VolumeVector<double> integral_gmm_volumes(n_classes);
            for (int j = 0; j < n_classes; ++j)
            {
                integral_gmm_volumes[j] = createVolume<double>(width, height, depth, spacing, origin);
                integral_gmm_volumes[j]->FillBuffer(0);
            }

            VolumePtr<unsigned int> integral_n_voxels = createVolume<unsigned int>(width, height, depth, spacing, origin);
            integral_n_voxels->FillBuffer(0);

            for (int z = 0; z < depth; ++z)
            {
                for (int y = 0; y < height; ++y)
                {
                    for (int x = 0; x < width; ++x)
                    {
                        itk::Index<3> index;
                        index[0] = x;
                        index[1] = y;
                        index[2] = z;

                        bool inside = true;
                        if (masks[i]->GetPixel(index) == 0) inside = false;

                        unsigned int cum_sum_n_voxels = (inside ? 1: 0);
                        std::vector<int> cum_sum_mr(n_mr_channels, 0);
                        std::vector<double> cum_sum_gmm(n_classes, 0);

                        if(inside)
                        {
                            Vector<double> X(n_mr_channels);
                            for (int c = 0; c < n_mr_channels; ++c)
                            {
                                const short v = mr_data[i][c]->GetPixel(index);
                                X(c) = static_cast<double>(v);
                                cum_sum_mr[c] = v;
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
                                const double v = prob[j] / normalizer;
                                gmm_volumes[j]->SetPixel(index, v);
                                cum_sum_gmm[j] = v;
                            }
                        }

#define ACCUMULATE(ox, oy, oz, op) \
    itk::Offset<3> offset; \
    offset[0] = ox; \
    offset[1] = oy; \
    offset[2] = oz; \
    cum_sum_n_voxels op##= integral_n_voxels->GetPixel(index + offset); \
    for (int c = 0; c < n_mr_channels; ++c)\
                        {\
    cum_sum_mr[c] op##= integral_mr_volumes[c]->GetPixel(index + offset); \
                    }\
    for (int c = 0; c < n_classes; ++c)\
                        {\
    cum_sum_gmm[c] op##= integral_gmm_volumes[c]->GetPixel(index + offset); \
                    }

                        if (x > 0)
                        {
                            ACCUMULATE(-1, 0, 0, +)
                        }
                        if (y > 0)
                        {
                            ACCUMULATE(0, -1, 0, +)
                        }
                        if (z > 0)
                        {
                            ACCUMULATE(0, 0, -1, +)
                        }
                        if (x > 0 && y > 0)
                        {
                            ACCUMULATE(-1, -1, 0, -)
                        }
                        if (y > 0 && z > 0)
                        {
                            ACCUMULATE(0, -1, -1, -)
                        }
                        if (x > 0 && z > 0)
                        {
                            ACCUMULATE(-1, 0, -1, -)
                        }
                        if (x > 0 && y > 0 && z > 0)
                        {
                            ACCUMULATE(-1, -1, -1, +)
                        }

                        for (int c = 0; c < n_mr_channels; ++c)
                        {
                            integral_mr_volumes[c]->SetPixel(index, cum_sum_mr[c]);
                        }

                        for (int c = 0; c < n_classes; ++c)
                        {
                            integral_gmm_volumes[c]->SetPixel(index, cum_sum_gmm[c]);
                        }

                        integral_n_voxels->SetPixel(index, cum_sum_n_voxels);

                    }
                }
            }

            data[i] = std::make_tuple(mr_data[i], gmm_volumes, masks[i], integral_mr_volumes, integral_gmm_volumes, integral_n_voxels);
        }

        std::vector<DataInstance> training_data;
        std::vector<int> labels;
        int n_data = 0;
        for (int i = 0; i < data.size(); ++i)
        {
            n_data += addTrainingInstance(training_data, labels, data[i], gts[i]);
        }


        const int max_box_size = vm["max_box_size"].as<int>();
        const auto factory = std::bind(createSpatialFeature, max_box_size);

        std::vector<int> class_counts(n_classes, 0);
        for (int label : labels) ++class_counts[label];
        std::vector<double> weights(n_classes);
        for (int i = 0; i < n_classes; ++i)
        {
            weights[i] = 1.0 / class_counts[i];
        }

        const double sample_rate = 1;

        const double training_time = timeit<std::chrono::minutes>([&]()
        {
            forest.train(training_data, labels, factory, weights, sample_rate);
        });

        std::cout << "Done Training in " << training_time << " minutes." << std::endl;

    }

    /* TESTING */

    {
        std::vector<int> n_tests_per_class(n_classes, 0);
        std::vector<int> n_corrects_per_class(n_classes, 0);
        std::vector<int> n_corrects_gmm_per_class(n_classes, 0);
        std::vector<int> true_labels, pred_labels, pred_gmm_labels;

        const double testing_time = timeit<std::chrono::minutes>([&]()
        {

            std::vector<fs::path> paths;
            for (; dir_iter != fs::directory_iterator(); ++dir_iter)
            {
                paths.push_back(dir_iter->path());
            }

            const int n_tests = paths.size();

            //            tbb::parallel_for(0,
            //                              n_tests,
            //                              [&](int i)
            //            {
            //      #pragma omp parallel for
            for (int i = 0; i < paths.size(); ++i)
            {
                const fs::path instance_path(paths[i]);

                std::cout << instance_path.string() << std::endl;
                const fs::path flair_path(instance_path / fs::path("VSD.Brain.XX.O.MR_Flair"));
                const fs::path t1_path(instance_path / fs::path("VSD.Brain.XX.O.MR_T1"));
                const fs::path t1c_path(instance_path / fs::path("VSD.Brain.XX.O.MR_T1c"));
                const fs::path t2_path(instance_path / fs::path("VSD.Brain.XX.O.MR_T2"));
                const fs::path gt_path(instance_path / fs::path("VSD.Brain_3more.XX.XX.OT"));

                const vector<fs::path> volume_paths =
                {
                    flair_path,
                    t1_path,
                    t1c_path,
                    t2_path
                };

                const VolumePtr<short> gt = readVolume<short>(findMHA(gt_path).string());

                int width, height, depth;
                std::tie(width, height, depth) = getVolumeDimension<short>(gt);
                const auto spacing = gt->GetSpacing();
                const auto origin = gt->GetOrigin();

                std::vector<double> prob(n_classes);
                std::vector<VolumePtr<double>> gmm_volumes(n_classes);
                for (int j = 0; j < n_classes; ++j)
                {
                    gmm_volumes[j] = createVolume<double>(width, height, depth, spacing, origin);
                }

                VolumeVector<short> volumes;
                for (const fs::path& p : volume_paths)
                {
                    volumes.push_back(readVolume<short>(findMHA(p).string()));
                }

                VolumePtr<unsigned char> mask = createMask<short>(volumes[0], 1);

                VolumeVector<int> integral_mr_volumes(n_mr_channels);
                for (int j = 0; j < n_mr_channels; ++j)
                {
                    integral_mr_volumes[j] = createVolume<int>(width, height, depth, spacing, origin);
                    integral_mr_volumes[j]->FillBuffer(0);
                }

                VolumeVector<double> integral_gmm_volumes(n_classes);
                for (int j = 0; j < n_classes; ++j)
                {
                    integral_gmm_volumes[j] = createVolume<double>(width, height, depth, spacing, origin);
                    integral_gmm_volumes[j]->FillBuffer(0);
                }

                VolumePtr<unsigned int> integral_n_voxels = createVolume<unsigned int>(width, height, depth, spacing, origin);
                integral_n_voxels->FillBuffer(0);

                VolumePtr<unsigned char> prediction_gmm = createVolume<unsigned char>(width, height, depth, spacing, origin);

                for (int z = 0; z < depth; ++z)
                {
                    for (int y = 0; y < height; ++y)
                    {
                        for (int x = 0; x < width; ++x)
                        {
                            itk::Index<3> index;
                            index[0] = x;
                            index[1] = y;
                            index[2] = z;

                            bool inside = true;
                            if (mask->GetPixel(index) == 0)
                            {
                                prediction_gmm->SetPixel(index, 0);
                                inside = false;
                            }

                            unsigned int cum_sum_n_voxels = (inside ? 1: 0);
                            std::vector<int> cum_sum_mr(n_mr_channels, 0);
                            std::vector<double> cum_sum_gmm(n_classes, 0);

                            if(inside)
                            {
                                Vector<double> X(n_mr_channels);
                                for (int c = 0; c < n_mr_channels; ++c)
                                {
                                    const short v = volumes[c]->GetPixel(index);
                                    X(c) = static_cast<double>(v);
                                    cum_sum_mr[c] = v;
                                }

                                double mx = -std::numeric_limits<double>::max();
                                unsigned char map_sol = -1;
                                double normalizer = 0;
                                for (int j = 0; j < n_classes; ++j)
                                {
                                    const double p = gmms[j].evaluate(X) * prior[j];
                                    prob[j] = p;
                                    normalizer += p;
                                    if (p > mx)
                                    {
                                        map_sol = j;
                                        mx = p;
                                    }
                                }

                                assert(0 <= map_sol && map_sol <= n_classes);
                                prediction_gmm->SetPixel(index, map_sol);

                                for (int j = 0; j < n_classes; ++j)
                                {
                                    gmm_volumes[j]->SetPixel(index, prob[j] / normalizer);
                                    cum_sum_gmm[j] = prob[j] / normalizer;
                                }

                            }
#define ACCUMULATE(ox, oy, oz, op) \
    itk::Offset<3> offset; \
    offset[0] = ox; \
    offset[1] = oy; \
    offset[2] = oz; \
    cum_sum_n_voxels op##= integral_n_voxels->GetPixel(index + offset); \
    for (int c = 0; c < n_mr_channels; ++c)\
                            {\
    cum_sum_mr[c] op##= integral_mr_volumes[c]->GetPixel(index + offset); \
                        }\
    for (int c = 0; c < n_classes; ++c)\
                            {\
    cum_sum_gmm[c] op##= integral_gmm_volumes[c]->GetPixel(index + offset); \
                        }

                            if (x > 0)
                            {
                                ACCUMULATE(-1, 0, 0, +)
                            }
                            if (y > 0)
                            {
                                ACCUMULATE(0, -1, 0, +)
                            }
                            if (z > 0)
                            {
                                ACCUMULATE(0, 0, -1, +)
                            }
                            if (x > 0 && y > 0)
                            {
                                ACCUMULATE(-1, -1, 0, -)
                            }
                            if (y > 0 && z > 0)
                            {
                                ACCUMULATE(0, -1, -1, -)
                            }
                            if (x > 0 && z > 0)
                            {
                                ACCUMULATE(-1, 0, -1, -)
                            }
                            if (x > 0 && y > 0 && z > 0)
                            {
                                ACCUMULATE(-1, -1, -1, +)
                            }

                            for (int c = 0; c < n_mr_channels; ++c)
                            {
                                integral_mr_volumes[c]->SetPixel(index, cum_sum_mr[c]);
                            }

                            for (int c = 0; c < n_classes; ++c)
                            {
                                integral_gmm_volumes[c]->SetPixel(index, cum_sum_gmm[c]);
                            }

                            integral_n_voxels->SetPixel(index, cum_sum_n_voxels);
                        }
                    }
                }

                const Instance ins = std::make_tuple(volumes, gmm_volumes, mask, integral_mr_volumes, integral_gmm_volumes, integral_n_voxels);
                VolumePtr<unsigned char> prediction = createVolume<unsigned char>(width, height, depth, spacing, origin);

                for (int z = 0; z < depth; ++z)
                {
                    for (int y = 0; y < height; ++y)
                    {
                        for (int x = 0; x < width; ++x)
                        {
                            itk::Index<3> index;
                            index[0] = x;
                            index[1] = y;
                            index[2] = z;
                            if (mask->GetPixel(index) == 0)
                            {
                                prediction->SetPixel(index, 0);
                                continue;
                            }

                            DataInstance data(ins, x, y, z);

                            const unsigned char pred = forest.predict(data);
                            const unsigned char pred_gmm = prediction_gmm->GetPixel(index);
                            const unsigned char true_label = gt->GetPixel(index);

                            assert(true_label <= 4 && pred <= 4 && pred_gmm <= 4);

                            #pragma omp critical
                            {
                                ++n_tests_per_class[true_label];
                                n_corrects_per_class[true_label] += (pred == true_label ? 1 : 0);
                                n_corrects_gmm_per_class[true_label] += (pred_gmm == true_label ? 1:0);
                                true_labels.push_back((int)true_label);
                                pred_labels.push_back((int)pred);
                                pred_gmm_labels.push_back((int)pred_gmm);

                            }

                            prediction->SetPixel(index, pred);
                        }
                    }
                }

                const string filename = "seg.mha";
                const string filename2 = "seg_gmm.mha";
                const fs::path save_path = fs::absolute(instance_path) / fs::path(filename);
                const fs::path save_path2 = fs::absolute(instance_path) / fs::path(filename2);

                writeVolume<unsigned char>(prediction, save_path.string());
                writeVolume<unsigned char>(prediction_gmm, save_path2.string());
                //            }
                //            );
                //        }
                //        );
            }
        });


        ofstream os1("true");
        ofstream os2("pred");
        ofstream os3("pred_gmm");

        os1 << VectorMapper<int>(&true_labels[0], true_labels.size());
        os2 << VectorMapper<int>(&pred_labels[0], pred_labels.size());
        os3 << VectorMapper<int>(&pred_gmm_labels[0], pred_gmm_labels.size());

        std::cout << "Done Testing in " << testing_time << " minutes." << std::endl;


        std::cout << "******** Accuracy **********\n";
        {
            const double n_corrects = std::accumulate(n_corrects_per_class.begin(), n_corrects_per_class.end(), 0);
            const double n_tests = std::accumulate(n_tests_per_class.begin(), n_tests_per_class.end(), 0);
            std::cout << "Overall: " << static_cast<double>(n_corrects) / n_tests << std::endl;

            for (int i = 0; i < n_classes; ++i)
            {
                std::cout << static_cast<double>(n_corrects_per_class[i]) / n_tests_per_class[i] << std::endl;
            }
        }

        std::cout << "******** GMM Accuracy **********\n";
        {
            const double n_corrects = std::accumulate(n_corrects_gmm_per_class.begin(), n_corrects_gmm_per_class.end(), 0);
            const double n_tests = std::accumulate(n_tests_per_class.begin(), n_tests_per_class.end(), 0);
            std::cout << "Overall: " << static_cast<double>(n_corrects) / n_tests << std::endl;

            for (int i = 0; i < n_classes; ++i)
            {
                std::cout << static_cast<double>(n_corrects_gmm_per_class[i]) / n_tests_per_class[i] << std::endl;
            }
        }

    }

    return 0;
}


