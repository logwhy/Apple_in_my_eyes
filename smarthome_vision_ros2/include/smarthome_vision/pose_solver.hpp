#pragma once

#include <map>
#include <opencv2/core.hpp>
#include <string>

#include "smarthome_vision/types.hpp"

namespace smarthome_vision
{

class PoseSolver
{
public:
  PoseSolver() = default;

  void set_camera(const CameraIntrinsics & camera);
  void set_class_size_map(const std::map<int, cv::Size2f> & class_size_map);
  void set_class_radius_map(const std::map<int, float> & class_radius_map);
  void set_pose_method(const std::string & pose_method);
  void set_circle_radius_scale(double circle_radius_scale);

  PoseResult solve(const Detection & det) const;

private:
  PoseResult solveBboxPnp(const Detection & det, cv::Size2f target_size) const;
  PoseResult solveCircleRadius(const Detection & det, float target_radius) const;

  CameraIntrinsics camera_;
  std::map<int, cv::Size2f> class_size_map_;
  std::map<int, float> class_radius_map_;
  std::string pose_method_ = "bbox_pnp";
  double circle_radius_scale_ = 1.0;
};

}  // namespace smarthome_vision
