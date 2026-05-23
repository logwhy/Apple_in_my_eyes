#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "smarthome_vision/inference_backend.hpp"
#include "smarthome_vision/types.hpp"

namespace smarthome_vision
{

class TRTDetector : public InferenceBackend
{
public:
  TRTDetector(
    const std::string & engine_path,
    int input_width,
    int input_height,
    float conf_thres,
    float score_thres,
    bool output_keypoints,
    bool use_cuda_preprocess);

  ~TRTDetector();

  std::vector<RawPrediction> infer(const cv::Mat & image) override;

private:
  std::vector<char> loadEngineFile(const std::string & path);

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

private:
  std::string engine_path_;
  int input_width_ = 640;
  int input_height_ = 640;
  float conf_thres_ = 0.25f;
  float score_thres_ = 0.25f;
  bool output_keypoints_ = false;
  bool use_cuda_preprocess_ = true;

  nvinfer1::IRuntime * runtime_ = nullptr;
  nvinfer1::ICudaEngine * engine_ = nullptr;
  nvinfer1::IExecutionContext * context_ = nullptr;

  void * device_buffers_[2] = {nullptr, nullptr};
  std::vector<float> host_output_;

  size_t input_numel_ = 0;
  size_t output_numel_ = 0;

  std::string input_name_;
  std::string output_name_;

  cudaStream_t stream_ = nullptr;
};

}  // namespace smarthome_vision
