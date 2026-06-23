#include "smarthome_vision/pose_solver.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace smarthome_vision
{

namespace
{

double computeReprojError(
  const std::vector<cv::Point3f> & object_points,
  const std::vector<cv::Point2f> & image_points,
  const cv::Mat & camera_matrix,
  const cv::Mat & dist_coeffs,
  const cv::Vec3d & rvec,
  const cv::Vec3d & tvec)
{
  std::vector<cv::Point2f> projected;
  cv::projectPoints(
    object_points,
    rvec,
    tvec,
    camera_matrix,
    dist_coeffs,
    projected);

  if (projected.size() != image_points.size() || projected.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  double total_err = 0.0;
  for (size_t i = 0; i < projected.size(); ++i) {
    total_err += cv::norm(projected[i] - image_points[i]);
  }
  return total_err / static_cast<double>(projected.size());
}

bool poseValid(const cv::Vec3d & tvec)
{
  if (!std::isfinite(tvec[0]) || !std::isfinite(tvec[1]) || !std::isfinite(tvec[2])) {
    return false;
  }

  // 目标应在相机前方
  if (tvec[2] <= 0.0) {
    return false;
  }

  return true;
}

}  // namespace

void PoseSolver::set_camera(const CameraIntrinsics & camera)
{
  camera_ = camera;
}

void PoseSolver::set_class_size_map(const std::map<int, cv::Size2f> & class_size_map)
{
  class_size_map_ = class_size_map;
}

void PoseSolver::set_class_radius_map(const std::map<int, float> & class_radius_map)
{
  class_radius_map_ = class_radius_map;
}

void PoseSolver::set_pose_method(const std::string & pose_method)
{
  if (pose_method == "circle_radius") {
    pose_method_ = pose_method;
  } else {
    pose_method_ = "bbox_pnp";
  }
}

void PoseSolver::set_circle_radius_scale(double circle_radius_scale)
{
  circle_radius_scale_ = circle_radius_scale > 0.0 ? circle_radius_scale : 1.0;
}

PoseResult PoseSolver::solveBboxPnp(const Detection & det, cv::Size2f target_size) const
{
  PoseResult result;

  const float w = target_size.width;
  const float h = target_size.height;

  if (w <= 0.0f || h <= 0.0f) {
    return result;
  }

  // det.corners 顺序必须严格是：
  // 0 = TL, 1 = TR, 2 = BR, 3 = BL
  std::vector<cv::Point3f> object_points = {
    {-w / 2.0f, -h / 2.0f, 0.0f},  // TL
    { w / 2.0f, -h / 2.0f, 0.0f},  // TR
    { w / 2.0f,  h / 2.0f, 0.0f},  // BR
    {-w / 2.0f,  h / 2.0f, 0.0f}   // BL
  };

  std::vector<cv::Point2f> image_points = {
    det.corners[0], det.corners[1], det.corners[2], det.corners[3]
  };

  cv::Vec3d best_rvec(0.0, 0.0, 0.0);
  cv::Vec3d best_tvec(0.0, 0.0, 0.0);
  double best_err = std::numeric_limits<double>::infinity();
  bool ok_any = false;

  auto trySolve = [&](int flag) {
    cv::Vec3d rvec, tvec;
    bool ok = cv::solvePnP(
      object_points,
      image_points,
      camera_.camera_matrix,
      camera_.dist_coeffs,
      rvec,
      tvec,
      false,
      flag);

    if (!ok) {
      return;
    }

    if (!poseValid(tvec)) {
      return;
    }

    const double err = computeReprojError(
      object_points,
      image_points,
      camera_.camera_matrix,
      camera_.dist_coeffs,
      rvec,
      tvec);

    if (err < best_err) {
      best_err = err;
      best_rvec = rvec;
      best_tvec = tvec;
      ok_any = true;
    }
  };

  // 先试更通用的方案
  trySolve(cv::SOLVEPNP_SQPNP);
  trySolve(cv::SOLVEPNP_ITERATIVE);

  // 只有严格正方形时才尝试 IPPE_SQUARE
  if (std::fabs(w - h) < 1e-6f) {
    trySolve(cv::SOLVEPNP_IPPE_SQUARE);
  }

  // 误差过大认为无效
  if (!ok_any || best_err > 8.0) {
    return result;
  }

  result.success = true;
  result.rvec = best_rvec;
  result.tvec = best_tvec;
  return result;
}

PoseResult PoseSolver::solveCircleRadius(const Detection & det, float target_radius) const
{
  PoseResult result;
  if (!det.has_bbox || det.bbox.width <= 0.0f || det.bbox.height <= 0.0f) {
    return result;
  }

  if (target_radius <= 0.0f || circle_radius_scale_ <= 0.0) {
    return result;
  }

  if (camera_.camera_matrix.rows < 3 || camera_.camera_matrix.cols < 3) {
    return result;
  }

  const double fx = camera_.camera_matrix.at<double>(0, 0);
  const double fy = camera_.camera_matrix.at<double>(1, 1);
  const double cx = camera_.camera_matrix.at<double>(0, 2);
  const double cy = camera_.camera_matrix.at<double>(1, 2);
  if (fx <= 0.0 || fy <= 0.0) {
    return result;
  }

  const double center_u = static_cast<double>(det.bbox.x) + static_cast<double>(det.bbox.width) * 0.5;
  const double center_v = static_cast<double>(det.bbox.y) + static_cast<double>(det.bbox.height) * 0.5;
  const double radius_px =
    (static_cast<double>(det.bbox.width) + static_cast<double>(det.bbox.height)) * 0.25 *
    circle_radius_scale_;
  if (radius_px <= 1.0) {
    return result;
  }

  double x_norm = (center_u - cx) / fx;
  double y_norm = (center_v - cy) / fy;
  if (!camera_.dist_coeffs.empty()) {
    std::vector<cv::Point2f> image_points{
      cv::Point2f(static_cast<float>(center_u), static_cast<float>(center_v))};
    std::vector<cv::Point2f> normalized_points;
    cv::undistortPoints(
      image_points,
      normalized_points,
      camera_.camera_matrix,
      camera_.dist_coeffs);
    if (!normalized_points.empty()) {
      x_norm = normalized_points[0].x;
      y_norm = normalized_points[0].y;
    }
  }

  const double focal_px = 0.5 * (fx + fy);
  const double z = static_cast<double>(target_radius) * focal_px / radius_px;
  cv::Vec3d tvec(x_norm * z, y_norm * z, z);
  if (!poseValid(tvec)) {
    return result;
  }

  result.success = true;
  result.tvec = tvec;
  result.rvec = cv::Vec3d(0.0, 0.0, 0.0);
  return result;
}

PoseResult PoseSolver::solve(const Detection & det) const
{
  PoseResult result;

  if (camera_.camera_matrix.empty()) {
    return result;
  }

  if (pose_method_ == "circle_radius") {
    auto radius_it = class_radius_map_.find(det.class_id);
    if (radius_it != class_radius_map_.end()) {
      return solveCircleRadius(det, radius_it->second);
    }
  }

  auto size_it = class_size_map_.find(det.class_id);
  if (size_it == class_size_map_.end()) {
    return result;
  }

  return solveBboxPnp(det, size_it->second);
}

}  // namespace smarthome_vision
