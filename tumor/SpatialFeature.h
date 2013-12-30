#ifndef SPATIAL_FEATURE_H
#define SPATIAL_FEATURE_H

#include <cassert>
#include <cmath>
#include <tuple>
#include <limits>
#include <util/general.h>
#include <util.h>

class SpatialFeature
{
public:

    enum FeatureType
    {
        IntensityDifference,
        MeanIntensityDifference,
        IntensityRangeAlongRay
    };

    SpatialFeature(int box_w, int box_h, int box_d, int n_channels = 9) :
        box_size(box_w),
        box_width(box_w),
        box_height(box_h),
        box_depth(box_d),
        c1(randInt(0, n_channels)),
        c2(randInt(0, n_channels)),
        offset_x(randInt(-box_size / 2, (box_size / 2) + 1)),
        offset_y(randInt(-box_size / 2, (box_size / 2) + 1)),
        offset_z(randInt(-box_size / 2, (box_size / 2) + 1))
    {
        const int n = randInt(0, 3);
        if (n == 0) feature = IntensityDifference;
        else if (n == 1) feature = MeanIntensityDifference;
        else feature = IntensityRangeAlongRay;
    }

    template <typename D>
    double operator()(const D& d) const
    {
        const int x = d.x;
        const int y = d.y;
        const int z = d.z;
        const VolumeVector<short>& vols = std::get<0>(d.instance);
        const VolumeVector<double>& gmms = std::get<1>(d.instance);
        const VolumePtr<unsigned char>& mask = std::get<2>(d.instance);
        int width, height, depth;
        std::tie(width, height, depth) = getVolumeDimension<unsigned char>(mask);
        const int dim[] = { width, height, depth };

        itk::Offset<3> offset;
        offset[0] = offset_x;
        offset[1] = offset_y;
        offset[2] = offset_z;

        const int n_mr_channels = vols.size();
        if (feature == IntensityDifference)
        {
            Index<short> index;
            index[0] = x;
            index[1] = y;
            index[2] = z;

            double v1;
            if (c1 < n_mr_channels) v1 = static_cast<double>(vols[c1]->GetPixel(index));
            else v1 = gmms[c1 - n_mr_channels]->GetPixel(index);

            double v2;
            if (outOfBounds<short>(width, height, depth, index + offset)) return v1;

            if (c2 < n_mr_channels) v2 = static_cast<double>(vols[c2]->GetPixel(index + offset));
            else v2 = gmms[c2 - n_mr_channels]->GetPixel(index + offset);

            return v1 - v2;
        }
        else if (feature == MeanIntensityDifference)
        {
            const VolumeVector<short>& integral_mrs = std::get<3>(d.instance);
            const VolumeVector<double>& integral_gmms = std::get<4>(d.instance);
            const VolumePtr<unsigned int>& integral_n_voxels = std::get<5>(d.instance);

            itk::Index<3> index1;
            index1[0] = x - box_size / 2;
            index1[1] = y - box_size / 2;
            index1[2] = z - box_size / 2;
            
            itk::Index<3> index2;
            index2[0] = x + box_size / 2;
            index2[1] = y + box_size / 2;
            index2[2] = z + box_size / 2;

            itk::Index<3> offset_index1 = index1 + offset;
            itk::Index<3> offset_index2 = index2 + offset;
            for (int i = 0; i < 3; ++i)
            {
                if (index1[i] < 0) index1[i] = 0;
                if (index2[i] >= dim[i]) index2[i] = dim[i] - 1;
                if (offset_index1[i] < 0) offset_index1[i] = 0;
                if (offset_index1[i] >= dim[i]) offset_index1[i] = dim[i] - 1;
                if (offset_index2[i] < 0) offset_index2[i] = 0;
                if (offset_index2[i] >= dim[i]) offset_index2[i] = dim[i] - 1;
                assert(index1[i] <= index2[i]);
                assert(offset_index1[i] <= offset_index2[i]);
            }

            double intensity1;
            if (c1 < n_mr_channels)
            {
                intensity1 = static_cast<double>(computeIntegral<short>(integral_mrs[c1], index1, index2));
            }
            else
            {
                intensity1 = computeIntegral<double>(integral_gmms[c1 - n_mr_channels], index1, index2);
            }
            const int n_voxels1 = computeIntegral<unsigned int>(integral_n_voxels, index1, index2);
            
            double intensity2;
            if (c2 < n_mr_channels)
            {
                intensity2 = static_cast<double>(computeIntegral<short>(integral_mrs[c2], offset_index1, offset_index2));
            }
            else
            {
                intensity2 = computeIntegral<double>(integral_gmms[c2 - n_mr_channels], offset_index1, offset_index2);
            }
            const int n_voxels2 = computeIntegral<unsigned int>(integral_n_voxels, offset_index1, offset_index2);

            return intensity1 / n_voxels1 - intensity2 / n_voxels2;
            //int n_voxels1 = 0;
            //double intensity1 = 0;
            //for (int zz = 0; zz < box_size; ++zz)
            //{
            //    for (int yy = 0; yy < box_size; ++yy)
            //    {
            //        for (int xx = 0; xx < box_size; ++xx)
            //        {
            //            Index<short> index;
            //            index[0] = x + xx - box_size / 2;
            //            index[1] = y + yy - box_size / 2;
            //            index[2] = z + zz - box_size / 2;

            //            if (outOfBounds<short>(width, height, depth, index)) continue;

            //            ++n_voxels1;
            //            if (c1 < n_mr_channels) intensity1 += static_cast<double>(vols[c1]->GetPixel(index));
            //            else intensity1 += gmms[c1 - n_mr_channels]->GetPixel(index);
            //        }
            //    }
            //}

            //intensity1 /= n_voxels1;
            //int n_voxels2 = 0;
            //double intensity2 = 0;
            //for (int zz = 0; zz < box_size; ++zz)
            //{
            //    for (int yy = 0; yy < box_size; ++yy)
            //    {
            //        for (int xx = 0; xx < box_size; ++xx)
            //        {
            //            Index<short> index;
            //            index[0] = offset_index[0] + xx - box_size / 2;
            //            index[1] = offset_index[1] + yy - box_size / 2;
            //            index[2] = offset_index[2] + zz - box_size / 2;

            //            if (outOfBounds<short>(width, height, depth, index)) continue;

            //            ++n_voxels2;
            //            if (c2 < n_mr_channels) intensity2 += static_cast<double>(vols[c2]->GetPixel(index));
            //            else intensity2 += gmms[c2 - n_mr_channels]->GetPixel(index);
            //        }
            //    }
            //}

            //intensity2 /= n_voxels2;

            //return intensity1 - intensity2;
        }
        else
        {
            double max_intensity = std::numeric_limits<double>::min();
            double min_intensity = std::numeric_limits<double>::max();
            const double lambda_incr = 0.1;

            for (double lambda = 0; lambda <= 1.0; lambda += lambda_incr)
            {
                Index<short> index;
                index[0] = x + std::round(lambda * offset_x);
                index[1] = y + std::round(lambda * offset_y);
                index[2] = z + std::round(lambda * offset_z);

                if (outOfBounds<short>(width, height, depth, index)) break;

                double v;
                if (c1 < n_mr_channels) v = static_cast<double>(vols[c1]->GetPixel(index));
                else v = gmms[c1 - n_mr_channels]->GetPixel(index);

                if (v > max_intensity) max_intensity = v;
                if (v < min_intensity) min_intensity = v;

            }

            return max_intensity - min_intensity;
        }
    }

private:
    int box_size;
    int box_width, box_height, box_depth;
    int c1, c2;
    int offset_x, offset_y, offset_z;
    FeatureType feature;
};

SpatialFeature createSpatialFeature(int max_box_size)
{
    const int min_box_size = 5;
    assert(min_box_size < max_box_size);
    return SpatialFeature(randInt(min_box_size, max_box_size + 1),
                          randInt(min_box_size, max_box_size + 1),
                          randInt(min_box_size, max_box_size + 1));
}

#endif
