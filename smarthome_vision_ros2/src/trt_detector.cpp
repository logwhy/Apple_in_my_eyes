#include "smarthome_vision/trt_detector.hpp"

#include <cuda_runtime.h>
#include <NvInfer.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace smarthome_vision
{

namespace
{

class TRTLogger : public nvinfer1::ILogger
{
public:
  void log(Severity severity, const char * msg) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      std::cerr << "[TensorRT] " << msg << std::endl;
    }
  }
};

TRTLogger g_logger;

inline size_t getSizeByDim(const nvinfer1::Dims & dims)
{
  size_t size = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      throw std::runtime_error("tensor shape still contains dynamic dimension");
    }
    size *= static_cast<size_t>(dims.d[i]);
  }
  return size;
}

inline void checkCuda(cudaError_t code, const std::string & msg)
{
  if (code != cudaSuccess) {
    throw std::runtime_error(msg + ": " + cudaGetErrorString(code));
  }
}

inline float clampf(float v, float lo, float hi)
{
  return std::max(lo, std::min(v, hi));
}

inline float iouRect(const cv::Rect2f & a, const cv::Rect2f & b)
{
  const float xx1 = std::max(a.x, b.x);
  const float yy1 = std::max(a.y, b.y);
  const float xx2 = std::min(a.x + a.width,  b.x + b.width);
  const float yy2 = std::min(a.y + a.height, b.y + b.height);

  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float uni = a.area() + b.area() - inter;

  if (uni <= 1e-6f) {
    return 0.0f;
  }
  return inter / uni;
}

std::vector<RawPrediction> applyNms(
  const std::vector<RawPrediction> & input,
  float iou_threshold)
{
  if (input.empty()) {
    return {};
  }

  std::vector<int> order(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    order[i] = static_cast<int>(i);
  }

  std::sort(order.begin(), order.end(),
    [&](int a, int b) {
      return input[a].score > input[b].score;
    });

  std::vector<RawPrediction> output;
  std::vector<bool> removed(input.size(), false);

  for (size_t oi = 0; oi < order.size(); ++oi) {
    const int i = order[oi];
    if (removed[i]) {
      continue;
    }

    output.push_back(input[i]);

    for (size_t oj = oi + 1; oj < order.size(); ++oj) {
      const int j = order[oj];
      if (removed[j]) {
        continue;
      }

      if (input[i].class_id != input[j].class_id) {
        continue;
      }

      if (iouRect(input[i].bbox, input[j].bbox) > iou_threshold) {
        removed[j] = true;
      }
    }
  }

  return output;
}

}  // namespace

TRTDetector::TRTDetector(
  const std::string & engine_path,
  int input_width,
  int input_height,
  float conf_thres,
  float score_thres,
  bool output_keypoints,
  bool use_cuda_preprocess)
: engine_path_(engine_path),
  input_width_(input_width),
  input_height_(input_height),
  conf_thres_(conf_thres),
  score_thres_(score_thres),
  output_keypoints_(output_keypoints),
  use_cuda_preprocess_(use_cuda_preprocess)
{
  if (engine_path_.empty()) {
    throw std::runtime_error("engine path is empty");
  }
  if (input_width_ <= 0 || input_height_ <= 0) {
    throw std::runtime_error("invalid input size");
  }

  auto engine_data = loadEngineFile(engine_path_);

  runtime_ = nvinfer1::createInferRuntime(g_logger);
  if (!runtime_) {
    throw std::runtime_error("failed to create TensorRT runtime");
  }

  engine_ = runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size());
  if (!engine_) {
    throw std::runtime_error("failed to deserialize engine: " + engine_path_);
  }

  context_ = engine_->createExecutionContext();
  if (!context_) {
    throw std::runtime_error("failed to create execution context");
  }

  const int nb = engine_->getNbIOTensors();
  for (int i = 0; i < nb; ++i) {
    const char * name = engine_->getIOTensorName(i);
    auto mode = engine_->getTensorIOMode(name);
    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = name;
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_name_ = name;
    }
  }

  if (input_name_.empty() || output_name_.empty()) {
    throw std::runtime_error("failed to find input/output tensor names");
  }

  auto input_dims = engine_->getTensorShape(input_name_.c_str());
  bool dynamic_input = false;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    if (input_dims.d[i] == -1) {
      dynamic_input = true;
      break;
    }
  }

  if (dynamic_input) {
    nvinfer1::Dims4 fixed_dims{1, 3, input_height_, input_width_};
    if (!context_->setInputShape(input_name_.c_str(), fixed_dims)) {
      throw std::runtime_error("failed to set dynamic input shape");
    }
    input_dims = context_->getTensorShape(input_name_.c_str());
  }

  auto output_dims = context_->getTensorShape(output_name_.c_str());
  for (int i = 0; i < output_dims.nbDims; ++i) {
    if (output_dims.d[i] < 0) {
      throw std::runtime_error("output tensor shape is still dynamic after setting input shape");
    }
  }

  std::cout << "\n========== 模型加载成功 ==========\n";
  std::cout << "Engine Path: " << engine_path_ << "\n";
  std::cout << "Output Shape: [ ";
  for (int i = 0; i < output_dims.nbDims; ++i) {
    std::cout << output_dims.d[i] << (i == output_dims.nbDims - 1 ? "" : ", ");
  }
  std::cout << " ]\n===================================\n\n";

  input_numel_ = getSizeByDim(input_dims);
  output_numel_ = getSizeByDim(output_dims);
  host_output_.resize(output_numel_);

  checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate failed");
  checkCuda(cudaMalloc(&device_buffers_[0], input_numel_ * sizeof(float)), "cudaMalloc input failed");
  checkCuda(cudaMalloc(&device_buffers_[1], output_numel_ * sizeof(float)), "cudaMalloc output failed");

  if (!context_->setTensorAddress(input_name_.c_str(), device_buffers_[0])) {
    throw std::runtime_error("failed to set input tensor address");
  }
  if (!context_->setTensorAddress(output_name_.c_str(), device_buffers_[1])) {
    throw std::runtime_error("failed to set output tensor address");
  }
}

TRTDetector::~TRTDetector()
{
  if (device_buffers_[0]) cudaFree(device_buffers_[0]);
  if (device_buffers_[1]) cudaFree(device_buffers_[1]);
  if (stream_) cudaStreamDestroy(stream_);
  if (context_) delete context_;
  if (engine_) delete engine_;
  if (runtime_) delete runtime_;
}

std::vector<char> TRTDetector::loadEngineFile(const std::string & path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("failed to open engine file: " + path);
  }

  file.seekg(0, std::ios::end);
  const size_t size = static_cast<size_t>(file.tellg());
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  file.read(buffer.data(), static_cast<std::streamsize>(size));

  if (!file) {
    throw std::runtime_error("failed to read engine file fully: " + path);
  }

  return buffer;
}

void TRTDetector::preprocess(
  const cv::Mat & image,
  std::vector<float> & input_tensor,
  float & scale_x,
  float & scale_y) const
{
  // 先保持你现在的直接 resize 逻辑
  // 若训练时使用的是 letterbox，后续再统一改预处理和反变换
  scale_x = static_cast<float>(image.cols) / static_cast<float>(input_width_);
  scale_y = static_cast<float>(image.rows) / static_cast<float>(input_height_);

  cv::Mat rgb_float;

  if (use_cuda_preprocess_) {
    cv::cuda::GpuMat gpu_bgr, gpu_resized, gpu_rgb;
    gpu_bgr.upload(image);
    cv::cuda::resize(gpu_bgr, gpu_resized, cv::Size(input_width_, input_height_));
    cv::cuda::cvtColor(gpu_resized, gpu_rgb, cv::COLOR_BGR2RGB);

    cv::Mat rgb;
    gpu_rgb.download(rgb);
    rgb.convertTo(rgb_float, CV_32F, 1.0 / 255.0);
  } else {
    cv::Mat resized, rgb;
    cv::resize(image, resized, cv::Size(input_width_, input_height_));
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb_float, CV_32F, 1.0 / 255.0);
  }

  input_tensor.resize(input_numel_);
  const int hw = input_width_ * input_height_;

  std::vector<cv::Mat> chw(3);
  for (int c = 0; c < 3; ++c) {
    chw[c] = cv::Mat(input_height_, input_width_, CV_32F, input_tensor.data() + c * hw);
  }
  cv::split(rgb_float, chw);
}

std::vector<RawPrediction> TRTDetector::infer(const cv::Mat & image)
{
  std::vector<RawPrediction> results;
  if (image.empty()) {
    return results;
  }

  float scale_x = 1.0f;
  float scale_y = 1.0f;
  std::vector<float> input_tensor;
  preprocess(image, input_tensor, scale_x, scale_y);

  checkCuda(
    cudaMemcpyAsync(
      device_buffers_[0],
      input_tensor.data(),
      input_numel_ * sizeof(float),
      cudaMemcpyHostToDevice,
      stream_),
    "cudaMemcpyAsync input failed");

  if (!context_->enqueueV3(stream_)) {
    throw std::runtime_error("TensorRT enqueueV3 failed");
  }

  checkCuda(
    cudaMemcpyAsync(
      host_output_.data(),
      device_buffers_[1],
      output_numel_ * sizeof(float),
      cudaMemcpyDeviceToHost,
      stream_),
    "cudaMemcpyAsync output failed");

  checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize failed");

  auto output_dims = context_->getTensorShape(output_name_.c_str());

  int num_preds = 0;
  int stride = 0;
  bool channels_first = false;

  if (output_dims.nbDims == 3) {
    const int a = output_dims.d[1];
    const int b = output_dims.d[2];
    if (a > 0 && b > 0 && (a == 6 || a == 18 || (a < b && a >= 5))) {
      stride = a;
      num_preds = b;
      channels_first = true;
    } else {
      num_preds = a;
      stride = b;
    }
  } else if (output_dims.nbDims == 2) {
    const int a = output_dims.d[0];
    const int b = output_dims.d[1];
    if (a > 0 && b > 0 && (a == 6 || a == 18 || (a < b && a >= 5))) {
      stride = a;
      num_preds = b;
      channels_first = true;
    } else {
      num_preds = a;
      stride = b;
    }
  } else {
    stride = output_keypoints_ ? 18 : 6;
    num_preds = static_cast<int>(output_numel_ / stride);
  }

  if (stride <= 0 || num_preds <= 0) {
    return {};
  }

  auto decoded = decode(
    host_output_.data(),
    num_preds,
    stride,
    channels_first,
    scale_x,
    scale_y,
    image.cols,
    image.rows);

  return applyNms(decoded, 0.45f);
}

std::vector<RawPrediction> TRTDetector::decode(
  const float * output,
  int num_preds,
  int stride,
  bool channels_first,
  float scale_x,
  float scale_y,
  int image_w,
  int image_h) const
{
  std::vector<RawPrediction> results;
  if (!output || num_preds <= 0 || stride <= 0) {
    return results;
  }

  auto at = [&](int pred_idx, int value_idx) -> float {
    if (channels_first) {
      return output[value_idx * num_preds + pred_idx];
    }
    return output[pred_idx * stride + value_idx];
  };

  // Supported layouts:
  // 1) NMS rows: [x1, y1, x2, y2, score, class_id] plus optional 4 keypoints.
  // 2) YOLOv8 bbox-only raw output: [cx, cy, w, h, cls0, cls1, ...].
  //    Many TensorRT exports expose this as [1, channels, anchors].
  const bool yolo_raw_bbox_only = !output_keypoints_ && stride > 6;

  for (int i = 0; i < num_preds; ++i) {
    int class_id = -1;
    float score = 0.0f;
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;

    if (yolo_raw_bbox_only) {
      float best_score = at(i, 4);
      int best_class = 0;
      for (int c = 5; c < stride; ++c) {
        const float cls_score = at(i, c);
        if (cls_score > best_score) {
          best_score = cls_score;
          best_class = c - 4;
        }
      }

      score = best_score;
      class_id = best_class;

      const float cx = at(i, 0) * scale_x;
      const float cy = at(i, 1) * scale_y;
      const float w = at(i, 2) * scale_x;
      const float h = at(i, 3) * scale_y;

      x1 = cx - w * 0.5f;
      y1 = cy - h * 0.5f;
      x2 = cx + w * 0.5f;
      y2 = cy + h * 0.5f;
    } else if (stride >= 6) {
      score = at(i, 4);
      class_id = static_cast<int>(std::lround(at(i, 5)));

      x1 = at(i, 0) * scale_x;
      y1 = at(i, 1) * scale_y;
      x2 = at(i, 2) * scale_x;
      y2 = at(i, 3) * scale_y;
    } else {
      continue;
    }
    if (score < score_thres_) {
      continue;
    }

    x1 = clampf(x1, 0.0f, static_cast<float>(image_w - 1));
    y1 = clampf(y1, 0.0f, static_cast<float>(image_h - 1));
    x2 = clampf(x2, 0.0f, static_cast<float>(image_w - 1));
    y2 = clampf(y2, 0.0f, static_cast<float>(image_h - 1));

    if (x2 <= x1 || y2 <= y1) {
      continue;
    }

    RawPrediction pred;
    pred.score = score;
    pred.class_id = class_id;
    pred.bbox = cv::Rect2f(x1, y1, x2 - x1, y2 - y1);

    if (output_keypoints_ && stride >= 18) {
      pred.has_keypoints = true;

      constexpr int kNumKpts = 4;
      constexpr int kKptBase = 6;
      constexpr int kKptStride = 3;

      std::array<float, kNumKpts> kpt_vis{};

      for (int k = 0; k < kNumKpts; ++k) {
        const int base = kKptBase + k * kKptStride;
        const float kx = at(i, base + 0) * scale_x;
        const float ky = at(i, base + 1) * scale_y;
        const float kv = at(i, base + 2);

        pred.keypoints[k] = cv::Point2f(kx, ky);
        kpt_vis[k] = kv;
      }

      int valid_kpt_count = 0;
      for (int k = 0; k < kNumKpts; ++k) {
        if (kpt_vis[k] >= conf_thres_) {
          ++valid_kpt_count;
        }
      }

      // All four keypoints are required when using keypoint corners.
      if (valid_kpt_count < kNumKpts) {
        continue;
      }
    } else {
      pred.has_keypoints = false;
    }

    results.push_back(pred);
  }

  return results;
}

}  // namespace smarthome_vision
