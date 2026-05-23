#pragma once

#include <opencv2/core.hpp>
#include <array>
#include <map>
#include <string>
#include <vector>

namespace smarthome_vision
{

enum class CornerSource
{
  KEYPOINT = 0,
  BBOX = 1
};

struct CameraIntrinsics
{
  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
};

struct RawPrediction
{
  int class_id = -1;
  float score = 0.0f;
  cv::Rect2f bbox;
  std::array<cv::Point2f, 4> keypoints{};
  bool has_keypoints = false;
};

struct Detection
{
  int class_id = -1;
  float score = 0.0f;

  cv::Rect2f bbox;
  std::array<cv::Point2f, 4> corners{};
  std::array<cv::Point2f, 4> keypoints{};

  bool has_bbox = false;
  bool has_keypoints = false;
  CornerSource corner_source = CornerSource::KEYPOINT;
};

struct PoseResult
{
  bool success = false;
  cv::Vec3d tvec{0.0, 0.0, 0.0};
  cv::Vec3d rvec{0.0, 0.0, 0.0};
};

}  // namespace smarthome_vision