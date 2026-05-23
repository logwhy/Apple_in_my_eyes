#pragma once

#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

#include "smarthome_vision/inference_backend.hpp"
#include "smarthome_vision/types.hpp"

namespace smarthome_vision
{

class Detector
{
public:
  Detector(
    const std::string & backend,
    const std::string & engine_path,
    const std::string & openvino_device,
    int input_width,
    int input_height,
    float conf_thres,
    float score_thres,
    bool output_keypoints,
    bool use_cuda_preprocess);

  std::vector<Detection> infer(const cv::Mat & image);

private:
  static std::array<cv::Point2f, 4> reorder_corners(
    const std::array<cv::Point2f, 4> & pts);

  static std::array<cv::Point2f, 4> bbox_to_corners(const cv::Rect2f & box);

  static bool keypoints_valid(
    const std::array<cv::Point2f, 4> & pts,
    const cv::Rect2f & box,
    int img_w,
    int img_h);

private:
  bool use_cuda_preprocess_ = true;
  std::unique_ptr<InferenceBackend> backend_;
};

}  // namespace smarthome_vision
