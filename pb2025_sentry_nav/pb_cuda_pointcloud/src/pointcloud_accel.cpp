#include "pb_cuda_pointcloud/pointcloud_accel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "pcl_conversions/pcl_conversions.h"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include "cuda_api.hpp"

namespace pb_cuda_pointcloud
{
namespace
{

inline bool pointCloudHasFloatField(
  const sensor_msgs::msg::PointCloud2 & msg, const std::string & name)
{
  return std::any_of(msg.fields.begin(), msg.fields.end(), [&](const auto & field) {
    return field.name == name && field.datatype == sensor_msgs::msg::PointField::FLOAT32;
  });
}

inline bool runtimeCudaEnabled(const BackendOptions & options)
{
  return options.enable && compiledWithCuda() && cudaDeviceAvailable(options.device_id);
}

inline bool canUseIntPointCount(size_t point_count)
{
  return point_count <= static_cast<size_t>(std::numeric_limits<int>::max());
}

inline bool getPointCloud2PointCount(const sensor_msgs::msg::PointCloud2 & msg, int & point_count)
{
  const uint64_t count = static_cast<uint64_t>(msg.width) * static_cast<uint64_t>(msg.height);
  if (count > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  point_count = static_cast<int>(count);
  return true;
}

}  // namespace

bool compiledWithCuda()
{
  return PB_CUDA_POINTCLOUD_HAS_CUDA == 1;
}

bool cudaDeviceAvailable(int device_id)
{
#if PB_CUDA_POINTCLOUD_HAS_CUDA
  return detail::cudaDeviceAvailableImpl(device_id);
#else
  (void)device_id;
  return false;
#endif
}

bool filterLivox(
  const livox_ros_driver2::msg::CustomMsg & msg, PointCloudXYZI & output, int n_scans,
  int point_filter_num, double blind, double det_range, const BackendOptions & options)
{
#if PB_CUDA_POINTCLOUD_HAS_CUDA
  if (!runtimeCudaEnabled(options) || msg.points.empty()) {
    return false;
  }
  if (!canUseIntPointCount(msg.points.size())) {
    return false;
  }

  std::vector<detail::RawLivoxPoint> input;
  input.reserve(msg.points.size());
  for (size_t i = 1; i < msg.points.size(); ++i) {
    const auto & point = msg.points[i];
    if (
      point.line >= n_scans ||
      !((point.tag & 0x30) == 0x10 || (point.tag & 0x30) == 0x00)) {
      continue;
    }
    const auto & previous = msg.points[i - 1];
    input.push_back(detail::RawLivoxPoint{
      point.offset_time, point.x, point.y, point.z, previous.x, previous.y, previous.z,
      point.reflectivity, point.tag, point.line});
  }
  if (input.empty()) {
    return false;
  }

  std::vector<detail::RawPointXYZI> raw_output(input.size());
  int output_count = 0;
  const bool ok = detail::cudaFilterLivoxImpl(
    input.data(), static_cast<int>(input.size()), raw_output.data(), &output_count, n_scans,
    std::max(point_filter_num, 1), static_cast<float>(blind * blind),
    static_cast<float>(det_range * det_range), options.device_id);
  if (!ok) {
    return false;
  }

  output.clear();
  output.reserve(output_count);
  for (int i = 0; i < output_count; ++i) {
    PointType point;
    point.x = raw_output[i].x;
    point.y = raw_output[i].y;
    point.z = raw_output[i].z;
    point.intensity = raw_output[i].intensity;
    point.curvature = raw_output[i].curvature;
    output.push_back(point);
  }
  return true;
#else
  (void)msg;
  (void)output;
  (void)n_scans;
  (void)point_filter_num;
  (void)blind;
  (void)det_range;
  (void)options;
  return false;
#endif
}

bool voxelDownsample(
  const PointCloudXYZI & input, PointCloudXYZI & output, float leaf_size,
  const BackendOptions & options)
{
  if (leaf_size <= 0.0f || input.empty()) {
    return false;
  }
  if (!canUseIntPointCount(input.size())) {
    return false;
  }

#if PB_CUDA_POINTCLOUD_HAS_CUDA
  if (!runtimeCudaEnabled(options)) {
    return false;
  }

  std::vector<detail::RawPointXYZI> raw_input;
  raw_input.reserve(input.size());
  for (const auto & point : input.points) {
    raw_input.push_back(
      detail::RawPointXYZI{point.x, point.y, point.z, point.intensity, point.curvature});
  }

  std::vector<detail::RawPointXYZI> raw_output(raw_input.size());
  int output_count = 0;
  const bool ok = detail::cudaVoxelDownsampleImpl(
    raw_input.data(), static_cast<int>(raw_input.size()), raw_output.data(), &output_count,
    leaf_size, options.device_id);
  if (!ok) {
    return false;
  }

  output.clear();
  output.reserve(output_count);
  for (int i = 0; i < output_count; ++i) {
    PointType point;
    point.x = raw_output[i].x;
    point.y = raw_output[i].y;
    point.z = raw_output[i].z;
    point.intensity = raw_output[i].intensity;
    point.curvature = raw_output[i].curvature;
    output.push_back(point);
  }
  return true;
#else
  (void)output;
  (void)options;
  return false;
#endif
}

bool voxelDownsampleXYZI(
  const pcl::PointCloud<pcl::PointXYZI> & input, pcl::PointCloud<pcl::PointXYZI> & output,
  float leaf_size, const BackendOptions & options)
{
  if (leaf_size <= 0.0f || input.empty()) {
    return false;
  }
  if (!canUseIntPointCount(input.size())) {
    return false;
  }

#if PB_CUDA_POINTCLOUD_HAS_CUDA
  if (!runtimeCudaEnabled(options)) {
    return false;
  }

  std::vector<detail::RawPointXYZI> raw_input;
  raw_input.reserve(input.size());
  for (const auto & point : input.points) {
    raw_input.push_back(detail::RawPointXYZI{point.x, point.y, point.z, point.intensity, 0.0f});
  }

  std::vector<detail::RawPointXYZI> raw_output(raw_input.size());
  int output_count = 0;
  const bool ok = detail::cudaVoxelDownsampleImpl(
    raw_input.data(), static_cast<int>(raw_input.size()), raw_output.data(), &output_count,
    leaf_size, options.device_id);
  if (!ok) {
    return false;
  }

  output.clear();
  output.reserve(output_count);
  for (int i = 0; i < output_count; ++i) {
    pcl::PointXYZI point;
    point.x = raw_output[i].x;
    point.y = raw_output[i].y;
    point.z = raw_output[i].z;
    point.intensity = raw_output[i].intensity;
    output.push_back(point);
  }
  return true;
#else
  (void)output;
  (void)options;
  return false;
#endif
}

bool transformPointCloudInPlace(
  const sensor_msgs::msg::PointCloud2 & input, sensor_msgs::msg::PointCloud2 & output,
  const Eigen::Matrix4f & transform, const std::string & output_frame,
  const BackendOptions & options)
{
  if (!pointCloudHasFloatField(input, "x") || !pointCloudHasFloatField(input, "y") ||
      !pointCloudHasFloatField(input, "z")) {
    return false;
  }
  if (!runtimeCudaEnabled(options)) {
    return false;
  }

  output = input;
  output.header.frame_id = output_frame;
  int point_count = 0;
  if (!getPointCloud2PointCount(input, point_count)) {
    return false;
  }
  if (point_count <= 0) {
    return true;
  }

#if PB_CUDA_POINTCLOUD_HAS_CUDA
  std::vector<float> x(point_count), y(point_count), z(point_count);
  int index = 0;
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(input, "x"), iter_y(input, "y"),
       iter_z(input, "z");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    x[index] = *iter_x;
    y[index] = *iter_y;
    z[index] = *iter_z;
    index++;
  }

  std::vector<float> out_x(point_count), out_y(point_count), out_z(point_count);
  Eigen::Matrix<float, 4, 4, Eigen::RowMajor> row_major = transform;
  const bool ok = detail::cudaTransformPointCloudImpl(
    x.data(), y.data(), z.data(), out_x.data(), out_y.data(), out_z.data(), point_count,
    row_major.data(), options.device_id);
  if (!ok) {
    return false;
  }

  index = 0;
  for (sensor_msgs::PointCloud2Iterator<float> iter_x(output, "x"), iter_y(output, "y"),
       iter_z(output, "z");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    *iter_x = out_x[index];
    *iter_y = out_y[index];
    *iter_z = out_z[index];
    index++;
  }
  return true;
#else
  return false;
#endif
}

bool pointCloudToLaserScan(
  const sensor_msgs::msg::PointCloud2 & cloud_msg, sensor_msgs::msg::LaserScan & scan_msg,
  double min_height, double max_height, double min_intensity, double max_intensity,
  double range_min, double range_max, const BackendOptions & options)
{
  if (
    !std::isfinite(scan_msg.angle_min) || !std::isfinite(scan_msg.angle_max) ||
    !std::isfinite(scan_msg.angle_increment) || scan_msg.angle_increment <= 0.0f ||
    !std::isfinite(range_min) || !std::isfinite(range_max) || range_min < 0.0 ||
    range_max <= range_min || scan_msg.angle_max <= scan_msg.angle_min) {
    return false;
  }
  if (
    !pointCloudHasFloatField(cloud_msg, "x") || !pointCloudHasFloatField(cloud_msg, "y") ||
    !pointCloudHasFloatField(cloud_msg, "z") ||
    !pointCloudHasFloatField(cloud_msg, "intensity")) {
    return false;
  }
  if (!runtimeCudaEnabled(options) || scan_msg.ranges.empty()) {
    return false;
  }

#if PB_CUDA_POINTCLOUD_HAS_CUDA
  int point_count = 0;
  if (!getPointCloud2PointCount(cloud_msg, point_count)) {
    return false;
  }
  if (point_count <= 0) {
    return true;
  }

  std::vector<float> x(point_count), y(point_count), z(point_count), intensity(point_count);
  int index = 0;
  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud_msg, "x"),
       iter_y(cloud_msg, "y"), iter_z(cloud_msg, "z"), iter_i(cloud_msg, "intensity");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
    x[index] = *iter_x;
    y[index] = *iter_y;
    z[index] = *iter_z;
    intensity[index] = *iter_i;
    index++;
  }

  return detail::cudaPointCloudToLaserScanImpl(
    x.data(), y.data(), z.data(), intensity.data(), point_count, scan_msg.ranges.data(),
    static_cast<int>(scan_msg.ranges.size()), static_cast<float>(scan_msg.angle_min),
    static_cast<float>(scan_msg.angle_max), static_cast<float>(scan_msg.angle_increment),
    static_cast<float>(min_height), static_cast<float>(max_height),
    static_cast<float>(min_intensity), static_cast<float>(max_intensity),
    static_cast<float>(range_min), static_cast<float>(range_max), options.device_id);
#else
  return false;
#endif
}

}  // namespace pb_cuda_pointcloud
