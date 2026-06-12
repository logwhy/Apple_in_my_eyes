#include "smarthome_vision/pose_solver.hpp"

#include <opencv2/calib3d.hpp>

#include <cmath>
#include <limits>
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

PoseResult PoseSolver::solve_by_center_size(const Detection & det) const
{
  PoseResult result;

  auto it = class_size_map_.find(det.class_id);
  if (it == class_size_map_.end()) {
    return result;
  }

  if (camera_.camera_matrix.empty() || !det.has_bbox) {
    return result;
  }

  const float object_w = it->second.width;
  const float object_h = it->second.height;
  const float pixel_w = det.bbox.width;
  const float pixel_h = det.bbox.height;

  if (object_w <= 0.0f || object_h <= 0.0f || pixel_w <= 1.0f || pixel_h <= 1.0f) {
    return result;
  }

  const double fx = camera_.camera_matrix.at<double>(0, 0);
  const double fy = camera_.camera_matrix.at<double>(1, 1);
  const double cx = camera_.camera_matrix.at<double>(0, 2);
  const double cy = camera_.camera_matrix.at<double>(1, 2);

  if (fx <= 0.0 || fy <= 0.0) {
    return result;
  }

  const double z_from_w = fx * static_cast<double>(object_w) / static_cast<double>(pixel_w);
  const double z_from_h = fy * static_cast<double>(object_h) / static_cast<double>(pixel_h);
  const double z = 0.5 * (z_from_w + z_from_h);

  if (!std::isfinite(z) || z <= 0.0) {
    return result;
  }

  const cv::Point2f center(
    det.bbox.x + det.bbox.width * 0.5f,
    det.bbox.y + det.bbox.height * 0.5f);

  double xn = (static_cast<double>(center.x) - cx) / fx;
  double yn = (static_cast<double>(center.y) - cy) / fy;

  if (!camera_.dist_coeffs.empty()) {
    std::vector<cv::Point2f> src{center};
    std::vector<cv::Point2f> undistorted;
    cv::undistortPoints(src, undistorted, camera_.camera_matrix, camera_.dist_coeffs);
    if (!undistorted.empty()) {
      xn = undistorted[0].x;
      yn = undistorted[0].y;
    }
  }

  result.success = true;
  result.tvec = cv::Vec3d(xn * z, yn * z, z);
  result.rvec = cv::Vec3d(0.0, 0.0, 0.0);
  return result;
}

PoseResult PoseSolver::solve(const Detection & det) const
{
  PoseResult result;

  auto it = class_size_map_.find(det.class_id);
  if (it == class_size_map_.end()) {
    return result;
  }

  if (camera_.camera_matrix.empty()) {
    return result;
  }

  const float w = it->second.width;
  const float h = it->second.height;

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

}  // namespace smarthome_vision
