#include "smarthome_vision/openvino_detector.hpp"

#include <openvino/core/preprocess/pre_post_process.hpp>  
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace smarthome_vision
{

namespace
{

constexpr size_t kMaxNmsCandidates = 300;

float clampf(float v, float lo, float hi)
{
  return std::max(lo, std::min(v, hi));
}

float iouRect(const cv::Rect2f & a, const cv::Rect2f & b)
{
  const float xx1 = std::max(a.x, b.x);
  const float yy1 = std::max(a.y, b.y);
  const float xx2 = std::min(a.x + a.width, b.x + b.width);
  const float yy2 = std::min(a.y + a.height, b.y + b.height);

  const float w = std::max(0.0f, xx2 - xx1);
  const float h = std::max(0.0f, yy2 - yy1);
  const float inter = w * h;
  const float uni = a.area() + b.area() - inter;

  return uni <= 1e-6f ? 0.0f : inter / uni;
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
      if (removed[j] || input[i].class_id != input[j].class_id) {
        continue;
      }

      if (iouRect(input[i].bbox, input[j].bbox) > iou_threshold) {
        removed[j] = true;
      }
    }
  }

  return output;
}

std::vector<RawPrediction> limitNmsCandidates(std::vector<RawPrediction> input)
{
  if (input.size() <= kMaxNmsCandidates) {
    return input;
  }

  auto cutoff = input.begin() + static_cast<std::ptrdiff_t>(kMaxNmsCandidates);
  std::nth_element(
    input.begin(), cutoff, input.end(),
    [](const RawPrediction & a, const RawPrediction & b) {
      return a.score > b.score;
    });
  input.resize(kMaxNmsCandidates);
  return input;
}

bool hasDynamicShape(const ov::PartialShape & shape)
{
  return shape.is_dynamic();
}

}  // namespace

OpenVINODetector::OpenVINODetector(
  const std::string & model_path,
  const std::string & device,
  int input_width,
  int input_height,
  float conf_thres,
  float score_thres,
  bool output_keypoints)
: model_path_(model_path),
  device_(device.empty() ? "CPU" : device),
  input_width_(input_width),
  input_height_(input_height),
  conf_thres_(conf_thres),
  score_thres_(score_thres),
  output_keypoints_(output_keypoints)
{
  if (model_path_.empty()) {
    throw std::runtime_error("OpenVINO model path is empty");
  }
  if (input_width_ <= 0 || input_height_ <= 0) {
    throw std::runtime_error("invalid OpenVINO input size");
  }

  auto model = core_.read_model(model_path_);
  if (model->inputs().size() != 1) {
    throw std::runtime_error("OpenVINO detector expects exactly one input");
  }
  if (model->outputs().empty()) {
    throw std::runtime_error("OpenVINO detector expects at least one output");
  }

  const auto input = model->input();
  const std::string input_name = input.get_any_name();
  if (hasDynamicShape(input.get_partial_shape())) {
    model->reshape({{input_name, ov::PartialShape{
      1, 3, input_height_, input_width_}}});
  }

  const ov::Shape model_input_shape = model->input().get_shape();
  if (model_input_shape.size() != 4) {
    throw std::runtime_error("OpenVINO detector input must be 4D NCHW");
  }
  if (model_input_shape[1] != 3) {
    throw std::runtime_error("OpenVINO detector input must have 3 channels in NCHW layout");
  }

  input_height_ = static_cast<int>(model_input_shape[2]);
  input_width_ = static_cast<int>(model_input_shape[3]);

  ov::preprocess::PrePostProcessor ppp(model);
  ppp.input()
    .tensor()
    .set_element_type(ov::element::u8)
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR)
    .set_spatial_dynamic_shape();
  ppp.input().model().set_layout(ov::Layout("NCHW"));
  ppp.input()
    .preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
    .scale(255.0f);
  model = ppp.build();

  compiled_model_ = core_.compile_model(
    model,
    device_,
    {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)});
  infer_request_ = compiled_model_.create_infer_request();
}

const float * OpenVINODetector::outputDataAsFloat(const ov::Tensor & tensor)
{
  const size_t count = tensor.get_size();
  if (tensor.get_element_type() == ov::element::f32) {
    return tensor.data<const float>();
  }
  if (tensor.get_element_type() == ov::element::f16) {
    const ov::float16 * data = tensor.data<const ov::float16>();
    output_buffer_.resize(count);
    for (size_t i = 0; i < count; ++i) {
      output_buffer_[i] = static_cast<float>(data[i]);
    }
    return output_buffer_.data();
  }
  throw std::runtime_error("OpenVINO detector output must be FP32 or FP16");
}

std::vector<RawPrediction> OpenVINODetector::infer(const cv::Mat & image)
{
  if (image.empty()) {
    return {};
  }
  if (image.type() != CV_8UC3) {
    throw std::runtime_error("OpenVINO detector expects CV_8UC3 BGR input");
  }

  if (!image.isContinuous()) {
    image.copyTo(contiguous_image_);
  }
  const cv::Mat & input_image = image.isContinuous() ? image : contiguous_image_;

  const float scale_x = static_cast<float>(input_image.cols) / static_cast<float>(input_width_);
  const float scale_y = static_cast<float>(input_image.rows) / static_cast<float>(input_height_);

  ov::Tensor tensor(
    ov::element::u8,
    ov::Shape{
      1,
      static_cast<size_t>(input_image.rows),
      static_cast<size_t>(input_image.cols),
      3},
    input_image.data);
  infer_request_.set_input_tensor(tensor);
  infer_request_.infer();

  const ov::Tensor output_tensor = infer_request_.get_output_tensor(0);
  const auto output_shape = output_tensor.get_shape();
  const float * output = outputDataAsFloat(output_tensor);
  const size_t output_count = output_tensor.get_size();

  int num_preds = 0;
  int stride = 0;
  bool channels_first = false;

  if (output_shape.size() == 3) {
    const int a = static_cast<int>(output_shape[1]);
    const int b = static_cast<int>(output_shape[2]);
    if (a > 0 && b > 0 && (a == 6 || a == 18 || (a < b && a >= 5))) {
      stride = a;
      num_preds = b;
      channels_first = true;
    } else {
      num_preds = a;
      stride = b;
    }
  } else if (output_shape.size() == 2) {
    const int a = static_cast<int>(output_shape[0]);
    const int b = static_cast<int>(output_shape[1]);
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
    num_preds = static_cast<int>(output_count / static_cast<size_t>(stride));
  }

  if (stride <= 0 || num_preds <= 0) {
    return {};
  }

  return applyNms(
    limitNmsCandidates(decode(
      output,
      num_preds,
      stride,
      channels_first,
      scale_x,
      scale_y,
      image.cols,
      image.rows)),
    0.45f);
}

std::vector<RawPrediction> OpenVINODetector::decode(
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
