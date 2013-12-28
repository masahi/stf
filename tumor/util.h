#ifndef TUMOR_UTIL
#define TUMOR_UTIL

#include <itkImage.h>
#include <vector>

template <typename T>
using Volume = itk::Image<T, 3>;

template <typename T>
using VolumePtr = typename Volume<T>::Pointer;

template <typename T>
using Index = typename Volume<T>::IndexType;

template <typename T>
using VolumeVector = std::vector<VolumePtr<T>>;

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

template <typename T>
bool outOfBounds(int width, int height, int depth, Index<T> index)
{
    const int x = index[0];
    const int y = index[1];
    const int z = index[2];

    return !(0 <= x && x < width && 0 <= y && y < height && 0 <= z && z < depth);
    
}
#endif
