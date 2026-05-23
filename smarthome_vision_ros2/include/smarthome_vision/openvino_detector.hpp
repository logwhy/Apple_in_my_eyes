#pragma once

#include <openvino/openvino.hpp>
#include <opencv2/core.hpp>

#include <array>
#include <string>
#include <vector>

#include "smarthome_vision/inference_backend.hpp"
#include "smarthome_vision/types.hpp"

namespace smarthome_vision
{

class OpenVINODetector : public InferenceBackend
{
public:
  OpenVINODetector(
    const std::string & model_path,
    const std::string & device,
    int input_width,
    int input_height,
    float conf_thres,
    float score_thres,
    bool output_keypoints);

  std::vector<RawPrediction> infer(const cv::Mat & image) override;

private:
  void preprocess(
    const cv::Mat & image,
    std::vector<float> & input_tensor,
    float & scale_x,
    float & scale_y) const;

  std::vector<RawPrediction> decode(
    const float * output,
    int num_preds,
    int stride,
    bool channels_first,
    float scale_x,
    float scale_y,
    int image_w,
    int image_h) const;

  std::vector<float> outputAsFloat(const ov::Tensor & tensor);

private:
  std::string model_path_;
  std::string device_ = "CPU";
  int input_width_ = 640;
  int input_height_ = 640;
  float conf_thres_ = 0.25f;
  float score_thres_ = 0.25f;
  bool output_keypoints_ = false;

  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
  ov::Shape input_shape_;
};

}  // namespace smarthome_vision
