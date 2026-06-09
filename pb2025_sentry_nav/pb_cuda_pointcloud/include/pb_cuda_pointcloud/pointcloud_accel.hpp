#ifndef PB_CUDA_POINTCLOUD__POINTCLOUD_ACCEL_HPP_
#define PB_CUDA_POINTCLOUD__POINTCLOUD_ACCEL_HPP_

#include <array>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <livox_ros_driver2/msg/custom_msg.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

namespace pb_cuda_pointcloud
{

struct BackendOptions
{
  bool enable = true;
  int device_id = 0;
  bool profile = false;
};

using PointType = pcl::PointXYZINormal;
using PointCloudXYZI = pcl::PointCloud<PointType>;

bool compiledWithCuda();
bool cudaDeviceAvailable(int device_id = 0);

bool filterLivox(
  const livox_ros_driver2::msg::CustomMsg & msg, PointCloudXYZI & output, int n_scans,
  int point_filter_num, double blind, double det_range, const BackendOptions & options);

bool voxelDownsample(
  const PointCloudXYZI & input, PointCloudXYZI & output, float leaf_size,
  const BackendOptions & options);

bool voxelDownsampleXYZI(
  const pcl::PointCloud<pcl::PointXYZI> & input, pcl::PointCloud<pcl::PointXYZI> & output,
  float leaf_size, const BackendOptions & options);

bool transformPointCloudInPlace(
  const sensor_msgs::msg::PointCloud2 & input, sensor_msgs::msg::PointCloud2 & output,
  const Eigen::Matrix4f & transform, const std::string & output_frame,
  const BackendOptions & options);

bool pointCloudToLaserScan(
  const sensor_msgs::msg::PointCloud2 & cloud_msg, sensor_msgs::msg::LaserScan & scan_msg,
  double min_height, double max_height, double min_intensity, double max_intensity,
  double range_min, double range_max, const BackendOptions & options);

struct KnnResult
{
  std::vector<std::array<int, 5>> indices;
  std::vector<std::array<float, 5>> squared_distances;
};

bool knn5(
  const PointCloudXYZI & query, const PointCloudXYZI & target, float max_range,
  KnnResult & result, const BackendOptions & options);

}  // namespace pb_cuda_pointcloud

#endif  // PB_CUDA_POINTCLOUD__POINTCLOUD_ACCEL_HPP_
