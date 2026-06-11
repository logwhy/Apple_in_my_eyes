#ifndef PB_CUDA_POINTCLOUD__CUDA_API_HPP_
#define PB_CUDA_POINTCLOUD__CUDA_API_HPP_

#include <cstdint>

namespace pb_cuda_pointcloud
{
namespace detail
{

struct RawLivoxPoint
{
  uint32_t offset_time;
  float x;
  float y;
  float z;
  float previous_x;
  float previous_y;
  float previous_z;
  uint8_t reflectivity;
  uint8_t tag;
  uint8_t line;
};

struct RawPointXYZI
{
  float x;
  float y;
  float z;
  float intensity;
  float curvature;
};

bool cudaDeviceAvailableImpl(int device_id);

bool cudaFilterLivoxImpl(
  const RawLivoxPoint * input, int point_count, RawPointXYZI * output, int * output_count,
  int n_scans, int point_filter_num, float blind_sq, float det_range_sq, int device_id);

bool cudaTransformPointCloudImpl(
  const float * input_x, const float * input_y, const float * input_z, float * output_x,
  float * output_y, float * output_z, int point_count, const float * matrix_row_major,
  int device_id);

bool cudaVoxelDownsampleImpl(
  const RawPointXYZI * input, int point_count, RawPointXYZI * output, int * output_count,
  float leaf_size, int device_id);

bool cudaPointCloudToLaserScanImpl(
  const float * input_x, const float * input_y, const float * input_z,
  const float * input_intensity, int point_count, float * ranges, int ranges_size,
  float angle_min, float angle_max, float angle_increment, float min_height, float max_height,
  float min_intensity, float max_intensity, float range_min, float range_max, int device_id);

}  // namespace detail
}  // namespace pb_cuda_pointcloud

#endif  // PB_CUDA_POINTCLOUD__CUDA_API_HPP_
