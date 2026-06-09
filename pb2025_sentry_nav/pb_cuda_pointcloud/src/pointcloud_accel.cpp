#include "pb_cuda_pointcloud/pointcloud_accel.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

#include "pcl_conversions/pcl_conversions.h"
#include "sensor_msgs/point_cloud2_iterator.hpp"

#include "cuda_api.hpp"

namespace pb_cuda_pointcloud
{
namespace
{

struct VoxelKey
{
  int x;
  int y;
  int z;

  bool operator==(const VoxelKey & other) const
  {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash
{
  std::size_t operator()(const VoxelKey & key) const
  {
    const auto hx = std::hash<int>{}(key.x * 73856093);
    const auto hy = std::hash<int>{}(key.y * 19349663);
    const auto hz = std::hash<int>{}(key.z * 83492791);
    return hx ^ (hy << 1) ^ (hz << 2);
  }
};

struct VoxelAccum
{
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double intensity = 0.0;
  double normal_x = 0.0;
  double normal_y = 0.0;
  double normal_z = 0.0;
  double curvature = 0.0;
  int count = 0;
};

inline bool pointCloudHasField(const sensor_msgs::msg::PointCloud2 & msg, const std::string & name)
{
  return std::any_of(msg.fields.begin(), msg.fields.end(), [&](const auto & field) {
    return field.name == name;
  });
}

inline bool runtimeCudaEnabled(const BackendOptions & options)
{
  return options.enable && compiledWithCuda() && cudaDeviceAvailable(options.device_id);
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

  std::vector<detail::RawLivoxPoint> input;
  input.reserve(msg.points.size());
  for (const auto & point : msg.points) {
    input.push_back(detail::RawLivoxPoint{
      point.offset_time, point.x, point.y, point.z, point.reflectivity, point.tag, point.line});
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
  if (!options.enable || leaf_size <= 0.0f || input.empty()) {
    return false;
  }

  std::unordered_map<VoxelKey, VoxelAccum, VoxelKeyHash> voxels;
  voxels.reserve(input.size());
  const float inverse_leaf = 1.0f / leaf_size;
  for (const auto & point : input.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }
    const VoxelKey key{
      static_cast<int>(std::floor(point.x * inverse_leaf)),
      static_cast<int>(std::floor(point.y * inverse_leaf)),
      static_cast<int>(std::floor(point.z * inverse_leaf))};
    auto & accum = voxels[key];
    accum.x += point.x;
    accum.y += point.y;
    accum.z += point.z;
    accum.intensity += point.intensity;
    accum.normal_x += point.normal_x;
    accum.normal_y += point.normal_y;
    accum.normal_z += point.normal_z;
    accum.curvature += point.curvature;
    accum.count++;
  }

  output.clear();
  output.reserve(voxels.size());
  for (const auto & item : voxels) {
    const auto & accum = item.second;
    if (accum.count <= 0) {
      continue;
    }
    const float inv_count = 1.0f / static_cast<float>(accum.count);
    PointType point;
    point.x = static_cast<float>(accum.x * inv_count);
    point.y = static_cast<float>(accum.y * inv_count);
    point.z = static_cast<float>(accum.z * inv_count);
    point.intensity = static_cast<float>(accum.intensity * inv_count);
    point.normal_x = static_cast<float>(accum.normal_x * inv_count);
    point.normal_y = static_cast<float>(accum.normal_y * inv_count);
    point.normal_z = static_cast<float>(accum.normal_z * inv_count);
    point.curvature = static_cast<float>(accum.curvature * inv_count);
    output.push_back(point);
  }
  return true;
}

bool voxelDownsampleXYZI(
  const pcl::PointCloud<pcl::PointXYZI> & input, pcl::PointCloud<pcl::PointXYZI> & output,
  float leaf_size, const BackendOptions & options)
{
  if (!options.enable || leaf_size <= 0.0f || input.empty()) {
    return false;
  }

  struct Accum
  {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double intensity = 0.0;
    int count = 0;
  };

  std::unordered_map<VoxelKey, Accum, VoxelKeyHash> voxels;
  voxels.reserve(input.size());
  const float inverse_leaf = 1.0f / leaf_size;
  for (const auto & point : input.points) {
    if (!std::isfinite(point.x) || !std::isfinite(point.y) || !std::isfinite(point.z)) {
      continue;
    }
    const VoxelKey key{
      static_cast<int>(std::floor(point.x * inverse_leaf)),
      static_cast<int>(std::floor(point.y * inverse_leaf)),
      static_cast<int>(std::floor(point.z * inverse_leaf))};
    auto & accum = voxels[key];
    accum.x += point.x;
    accum.y += point.y;
    accum.z += point.z;
    accum.intensity += point.intensity;
    accum.count++;
  }

  output.clear();
  output.reserve(voxels.size());
  for (const auto & item : voxels) {
    const auto & accum = item.second;
    if (accum.count <= 0) {
      continue;
    }
    const float inv_count = 1.0f / static_cast<float>(accum.count);
    pcl::PointXYZI point;
    point.x = static_cast<float>(accum.x * inv_count);
    point.y = static_cast<float>(accum.y * inv_count);
    point.z = static_cast<float>(accum.z * inv_count);
    point.intensity = static_cast<float>(accum.intensity * inv_count);
    output.push_back(point);
  }
  return true;
}

bool transformPointCloudInPlace(
  const sensor_msgs::msg::PointCloud2 & input, sensor_msgs::msg::PointCloud2 & output,
  const Eigen::Matrix4f & transform, const std::string & output_frame,
  const BackendOptions & options)
{
  if (!pointCloudHasField(input, "x") || !pointCloudHasField(input, "y") ||
      !pointCloudHasField(input, "z")) {
    return false;
  }

  output = input;
  output.header.frame_id = output_frame;
  const int point_count = static_cast<int>(input.width * input.height);
  if (point_count <= 0) {
    return true;
  }

#if PB_CUDA_POINTCLOUD_HAS_CUDA
  if (runtimeCudaEnabled(options)) {
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
    if (ok) {
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
    }
  }
#else
  (void)options;
#endif

  sensor_msgs::PointCloud2ConstIterator<float> in_x(input, "x");
  sensor_msgs::PointCloud2ConstIterator<float> in_y(input, "y");
  sensor_msgs::PointCloud2ConstIterator<float> in_z(input, "z");
  sensor_msgs::PointCloud2Iterator<float> out_x(output, "x");
  sensor_msgs::PointCloud2Iterator<float> out_y(output, "y");
  sensor_msgs::PointCloud2Iterator<float> out_z(output, "z");
  for (; in_x != in_x.end(); ++in_x, ++in_y, ++in_z, ++out_x, ++out_y, ++out_z) {
    const Eigen::Vector4f p(*in_x, *in_y, *in_z, 1.0f);
    const Eigen::Vector4f out = transform * p;
    *out_x = out.x();
    *out_y = out.y();
    *out_z = out.z();
  }
  return true;
}

bool pointCloudToLaserScan(
  const sensor_msgs::msg::PointCloud2 & cloud_msg, sensor_msgs::msg::LaserScan & scan_msg,
  double min_height, double max_height, double min_intensity, double max_intensity,
  double range_min, double range_max, const BackendOptions & options)
{
  (void)options;
  if (
    !pointCloudHasField(cloud_msg, "x") || !pointCloudHasField(cloud_msg, "y") ||
    !pointCloudHasField(cloud_msg, "z") || !pointCloudHasField(cloud_msg, "intensity")) {
    return false;
  }

  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud_msg, "x"),
       iter_y(cloud_msg, "y"), iter_z(cloud_msg, "z"), iter_i(cloud_msg, "intensity");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
    if (std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z)) {
      continue;
    }
    if (*iter_z > max_height || *iter_z < min_height) {
      continue;
    }
    if (*iter_i < min_intensity || *iter_i > max_intensity) {
      continue;
    }
    const double range = std::hypot(*iter_x, *iter_y);
    if (range < range_min || range > range_max) {
      continue;
    }
    const double angle = std::atan2(*iter_y, *iter_x);
    if (angle < scan_msg.angle_min || angle > scan_msg.angle_max) {
      continue;
    }
    const int index = static_cast<int>((angle - scan_msg.angle_min) / scan_msg.angle_increment);
    if (index >= 0 && static_cast<size_t>(index) < scan_msg.ranges.size() &&
        range < scan_msg.ranges[index]) {
      scan_msg.ranges[index] = range;
    }
  }
  return true;
}

bool knn5(
  const PointCloudXYZI & query, const PointCloudXYZI & target, float max_range,
  KnnResult & result, const BackendOptions & options)
{
  (void)options;
  result.indices.assign(query.size(), std::array<int, 5>{-1, -1, -1, -1, -1});
  const float max_range_sq = max_range * max_range;
  result.squared_distances.assign(
    query.size(), std::array<float, 5>{
                    max_range_sq, max_range_sq, max_range_sq, max_range_sq, max_range_sq});
  if (query.empty() || target.empty()) {
    return false;
  }

  for (size_t qi = 0; qi < query.size(); ++qi) {
    for (size_t ti = 0; ti < target.size(); ++ti) {
      const float dx = query[qi].x - target[ti].x;
      const float dy = query[qi].y - target[ti].y;
      const float dz = query[qi].z - target[ti].z;
      const float dist_sq = dx * dx + dy * dy + dz * dz;
      if (dist_sq >= result.squared_distances[qi][4]) {
        continue;
      }
      int insert_at = 4;
      while (insert_at > 0 && dist_sq < result.squared_distances[qi][insert_at - 1]) {
        result.squared_distances[qi][insert_at] = result.squared_distances[qi][insert_at - 1];
        result.indices[qi][insert_at] = result.indices[qi][insert_at - 1];
        insert_at--;
      }
      result.squared_distances[qi][insert_at] = dist_sq;
      result.indices[qi][insert_at] = static_cast<int>(ti);
    }
  }
  return true;
}

}  // namespace pb_cuda_pointcloud
