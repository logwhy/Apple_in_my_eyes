#pragma once

#include <map>
#include <opencv2/core.hpp>

#include "smarthome_vision/types.hpp"

namespace smarthome_vision
{

class PoseSolver
{
public:
  PoseSolver() = default;

  void set_camera(const CameraIntrinsics & camera);
  void set_class_size_map(const std::map<int, cv::Size2f> & class_size_map);

  PoseResult solve(const Detection & det) const;

private:
  CameraIntrinsics camera_;
  std::map<int, cv::Size2f> class_size_map_;
};

}  // namespace smarthome_vision