#ifndef TUMOR_UTIL
#define TUMOR_UTIL

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkBinaryThresholdImageFilter.h>
#include <string>
#include <limits>
#include <vector>

template <typename T>
using Volume = itk::Image<T, 3>;

template <typename T>
using VolumePtr = typename Volume<T>::Pointer;

template <typename T>
using VolumeVector = std::vector<VolumePtr<T>>;

template <typename T>
T computeIntegral(const VolumePtr<T>& vol, const itk::Index<3>& index1, const itk::Index<3>& index2)
{
    T integral = vol->GetPixel(index2);

    itk::Index<3> index;
    index[0] = index1[0] - 1;
    index[1] = index1[1] -1 ;
    index[2] = index1[2] - 1;
    if(index[0] >= 0 && index[1] >= 0 && index[2] >= 0) vol->GetPixel(index);

    index[0] = index2[0];
    index[1] = index2[1];
    index[2] = index1[2] - 1;
    if(index[2] >= 0) integral -= vol->GetPixel(index);

    index[0] = index2[0];
    index[1] = index1[1] - 1;
    index[2] = index2[2];
    if(index[1] >= 0) integral -= vol->GetPixel(index);
    
    index[0] = index1[0] - 1;
    index[1] = index2[1];
    index[2] = index2[2];
    if(index[0] >= 0) integral -= vol->GetPixel(index);
    
    index[0] = index1[0] - 1;
    index[1] = index1[1] - 1;
    index[2] = index2[2];
    if(index[0] >= 0 && index[1] >= 0) integral += vol->GetPixel(index);

    index[0] = index1[0] - 1;
    index[1] = index2[1];
    index[2] = index1[2] - 1;
    if(index[0] >= 0 && index[2] >= 0) integral += vol->GetPixel(index);

    index[0] = index2[0];
    index[1] = index1[1]-1;
    index[2] = index1[2]-1;
    if(index[1] >= 0 && index[2] >= 0) integral += vol->GetPixel(index);

    return integral;
}

template <typename T>
VolumePtr<T> readVolume(const std::string& filename)
{
    typedef itk::ImageFileReader<Volume<T>> ImageReader;
    typename ImageReader::Pointer reader = ImageReader::New();
    reader->SetFileName(filename);
    reader->Update();
    return reader->GetOutput();
}

template <typename T>
VolumePtr<unsigned char> createMask(const VolumePtr<T>& volume,
                                    T lower_thres = std::numeric_limits<T>::min(),
                                    T upper_thres = std::numeric_limits<T>::max())
{
    typedef itk::BinaryThresholdImageFilter<Volume<T>, Volume<unsigned char>> ThresholdFilter;
    typename ThresholdFilter::Pointer thres = ThresholdFilter::New();
    thres->SetInput(volume);
    thres->SetLowerThreshold(lower_thres);
    thres->SetUpperThreshold(upper_thres);
    thres->SetInsideValue(1);
    thres->SetOutsideValue(0);
    thres->Update();
    return thres->GetOutput();
}

template <typename T>
void writeVolume(const VolumePtr<T>& volume, const std::string& filename)
{
    typedef itk::ImageFileWriter<Volume<T>> Writer;
    typename Writer::Pointer writer = Writer::New();
    writer->SetInput(volume);
    writer->SetFileName(filename);
    writer->Update();
}

template <typename T>
VolumePtr<T> createVolume(int width, int height, int depth, const itk::Vector<double, 3>& spacing, const itk::Point<double, 3>& origin)
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
    vol->SetSpacing(spacing);
    vol->SetOrigin(origin);
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

bool outOfBounds(int width, int height, int depth, itk::Index<3> index)
{
    const int x = index[0];
    const int y = index[1];
    const int z = index[2];

    return !(0 <= x && x < width && 0 <= y && y < height && 0 <= z && z < depth);
    
}
#endif
