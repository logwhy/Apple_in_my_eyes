#include "smarthome_vision/detector.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <vector>

namespace smarthome_vision
{

Detector::Detector(
  const std::string & engine_path,
  int input_width,
  int input_height,
  float conf_thres,
  float score_thres,
  bool output_keypoints,
  bool use_cuda_preprocess)
: use_cuda_preprocess_(use_cuda_preprocess)
{
  trt_detector_ = std::make_unique<TRTDetector>(
    engine_path,
    input_width,
    input_height,
    conf_thres,
    score_thres,
    output_keypoints,
    use_cuda_preprocess_);
}

// Keypoint order is expected to be TL, TR, BR, BL.
std::array<cv::Point2f, 4> Detector::reorder_corners(
  const std::array<cv::Point2f, 4> & pts)
{
  return pts;
}

std::array<cv::Point2f, 4> Detector::bbox_to_corners(const cv::Rect2f & box)
{
  return {
    cv::Point2f(box.x, box.y),
    cv::Point2f(box.x + box.width, box.y),
    cv::Point2f(box.x + box.width, box.y + box.height),
    cv::Point2f(box.x, box.y + box.height)
  };
}

bool Detector::keypoints_valid(
  const std::array<cv::Point2f, 4> & pts,
  const cv::Rect2f & box,
  int img_w,
  int img_h)
{
  for (const auto & p : pts) {
    if (p.x < 0.0f || p.y < 0.0f ||
        p.x >= static_cast<float>(img_w) ||
        p.y >= static_cast<float>(img_h))
    {
      return false;
    }
  }

  int outside_count = 0;
  for (const auto & p : pts) {
    if (!box.contains(p)) {
      outside_count++;
    }
  }

  if (outside_count > 1) {
    return false;
  }

  std::vector<cv::Point2f> poly = {pts[0], pts[1], pts[2], pts[3]};
  const float area = std::fabs(static_cast<float>(cv::contourArea(poly)));
  if (area < 25.0f) {
    return false;
  }

  return true;
}

std::vector<Detection> Detector::infer(const cv::Mat & image)
{
  std::vector<Detection> out;
  if (!trt_detector_) {
    return out;
  }

  const std::vector<RawPrediction> preds = trt_detector_->infer(image);

  for (const auto & pred : preds) {
    if (pred.bbox.width <= 0.0f || pred.bbox.height <= 0.0f) {
      continue;
    }

    Detection det;
    det.class_id = pred.class_id;
    det.score = pred.score;
    det.bbox = pred.bbox;
    det.has_bbox = true;

    if (pred.has_keypoints) {
      if (!keypoints_valid(pred.keypoints, pred.bbox, image.cols, image.rows)) {
        continue;
      }

      det.keypoints = pred.keypoints;
      det.has_keypoints = true;
      det.corners = reorder_corners(pred.keypoints);
      det.corner_source = CornerSource::KEYPOINT;
    } else {
      det.has_keypoints = false;
      det.corners = bbox_to_corners(pred.bbox);
      det.corner_source = CornerSource::BBOX;
    }

    out.push_back(det);
  }

  return out;
}

}  // namespace smarthome_vision
