#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "smarthome_vision/types.hpp"

namespace smarthome_vision
{

class InferenceBackend
{
public:
  virtual ~InferenceBackend() = default;

  virtual std::vector<RawPrediction> infer(const cv::Mat & image) = 0;
};

}  // namespace smarthome_vision
