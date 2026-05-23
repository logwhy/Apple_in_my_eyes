#include <memory>
#include <map>
#include <string>
#include <vector>
#include <array>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <std_msgs/msg/u_int8.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include "smarthome_vision/msg/detected_target.hpp"
#include "smarthome_vision/detector.hpp"
#include "smarthome_vision/pose_solver.hpp"
#include "smarthome_vision/types.hpp"
#include "smarthome_vision/protocol.hpp"

using std::placeholders::_1;

namespace smarthome_vision
{

namespace
{

cv::Scalar colorForClass(int class_id)
{
  static const std::array<cv::Scalar, 8> colors = {
    cv::Scalar(255, 0, 0),
    cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 255),
    cv::Scalar(255, 0, 255),
    cv::Scalar(255, 255, 0),
    cv::Scalar(0, 128, 255),
    cv::Scalar(128, 0, 255),
    cv::Scalar(255, 128, 0)
  };
  return colors[static_cast<size_t>(std::abs(class_id)) % colors.size()];
}

std::string modeToString(uint8_t mode)
{
  switch (static_cast<VisionMode>(mode)) {
    case VisionMode::IDLE:
      return "IDLE";
    case VisionMode::DETECT_OBJECT:
      return "OBJECT";
    case VisionMode::DETECT_QR:
      return "QR";
    default:
      return "UNKNOWN";
  }
}

int remapClassId(int raw_id, const std::vector<int> & mapping)
{
  if (raw_id >= 0 && static_cast<size_t>(raw_id) < mapping.size()) {
    return mapping[static_cast<size_t>(raw_id)];
  }
  return raw_id;
}

bool classAllowed(int class_id, const std::vector<int> & allowed_class_ids)
{
  if (allowed_class_ids.empty()) {
    return true;
  }
  return std::find(allowed_class_ids.begin(), allowed_class_ids.end(), class_id) !=
         allowed_class_ids.end();
}

std::map<int, cv::Size2f> buildSizeMap(
  const std::vector<int> & class_ids,
  const std::vector<double> & class_sizes)
{
  std::map<int, cv::Size2f> out;
  for (size_t i = 0; i < class_ids.size(); ++i) {
    const size_t idx = i * 2;
    if (idx + 1 < class_sizes.size()) {
      out[class_ids[i]] = cv::Size2f(
        static_cast<float>(class_sizes[idx]),
        static_cast<float>(class_sizes[idx + 1]));
    }
  }
  return out;
}

std::map<int, cv::Size2f> buildSquareSizeMap(
  const std::vector<int> & class_ids,
  float side_len)
{
  std::map<int, cv::Size2f> out;
  for (const int id : class_ids) {
    out[id] = cv::Size2f(side_len, side_len);
  }
  return out;
}

void drawDetectionDebug(cv::Mat & image, const Detection & det, const std::string & prefix)
{
  const cv::Scalar bbox_color = colorForClass(det.class_id);

  if (det.has_bbox) {
    cv::rectangle(image, det.bbox, bbox_color, 2);
  }

  std::ostringstream oss;
  oss << prefix
      << " ID:" << det.class_id
      << " Conf:" << std::fixed << std::setprecision(2) << det.score;

  const std::string label = oss.str();

  int baseline = 0;
  const cv::Size text_size =
    cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);

  const int tx = std::max(0, static_cast<int>(det.bbox.x));
  const int ty = std::max(text_size.height + 6, static_cast<int>(det.bbox.y) - 6);

  cv::rectangle(
    image,
    cv::Rect(tx, ty - text_size.height - 6, text_size.width + 8, text_size.height + 8),
    bbox_color,
    cv::FILLED);

  cv::putText(
    image,
    label,
    cv::Point(tx + 4, ty - 4),
    cv::FONT_HERSHEY_SIMPLEX,
    0.55,
    cv::Scalar(255, 255, 255),
    1);

  static const std::array<std::string, 4> names = {"TL", "TR", "BR", "BL"};

  for (int i = 0; i < 4; ++i) {
    const cv::Point2f & p = det.corners[i];
    cv::circle(image, p, 5, cv::Scalar(0, 0, 255), -1);
    cv::putText(
      image,
      names[i],
      cv::Point(static_cast<int>(p.x) + 6, static_cast<int>(p.y) - 6),
      cv::FONT_HERSHEY_SIMPLEX,
      0.5,
      cv::Scalar(0, 0, 255),
      2);
  }

  for (int i = 0; i < 4; ++i) {
    const cv::Point2f & p1 = det.corners[i];
    const cv::Point2f & p2 = det.corners[(i + 1) % 4];
    cv::line(image, p1, p2, cv::Scalar(0, 255, 255), 2);
  }
}

void drawPoseInsideBox(
  cv::Mat & image,
  const Detection & det,
  const PoseResult & pose)
{
  if (!det.has_bbox) {
    return;
  }

  const int x = std::max(0, static_cast<int>(det.bbox.x));
  const int y = std::max(0, static_cast<int>(det.bbox.y));

  const double font_scale = 0.45;
  const int thickness = 1;
  const int line_gap = 4;
  const int padding = 4;

  std::ostringstream sx, sy, sz;
  sx << "X=" << std::fixed << std::setprecision(3) << pose.tvec[0];
  sy << "Y=" << std::fixed << std::setprecision(3) << pose.tvec[1];
  sz << "Z=" << std::fixed << std::setprecision(3) << pose.tvec[2];

  const std::string line1 = sx.str();
  const std::string line2 = sy.str();
  const std::string line3 = sz.str();

  int base1 = 0, base2 = 0, base3 = 0;
  const cv::Size s1 = cv::getTextSize(line1, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &base1);
  const cv::Size s2 = cv::getTextSize(line2, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &base2);
  const cv::Size s3 = cv::getTextSize(line3, cv::FONT_HERSHEY_SIMPLEX, font_scale, thickness, &base3);

  const int text_w = std::max({s1.width, s2.width, s3.width});
  const int line_h = std::max({s1.height, s2.height, s3.height});
  const int box_h = padding * 2 + line_h * 3 + line_gap * 2;
  const int box_w = padding * 2 + text_w;

  int bx = x + 2;
  int by = y + 2;

  bx = std::min(std::max(0, bx), std::max(0, image.cols - box_w - 1));
  by = std::min(std::max(0, by), std::max(0, image.rows - box_h - 1));

  cv::rectangle(
    image,
    cv::Rect(bx, by, std::min(box_w, image.cols - bx), std::min(box_h, image.rows - by)),
    cv::Scalar(0, 0, 0),
    cv::FILLED);

  cv::rectangle(
    image,
    cv::Rect(bx, by, std::min(box_w, image.cols - bx), std::min(box_h, image.rows - by)),
    cv::Scalar(255, 255, 0),
    1);

  const int tx = bx + padding;
  int ty = by + padding + line_h;

  cv::putText(
    image, line1,
    cv::Point(tx, ty),
    cv::FONT_HERSHEY_SIMPLEX, font_scale,
    cv::Scalar(255, 255, 0), thickness);

  ty += line_h + line_gap;
  cv::putText(
    image, line2,
    cv::Point(tx, ty),
    cv::FONT_HERSHEY_SIMPLEX, font_scale,
    cv::Scalar(255, 255, 0), thickness);

  ty += line_h + line_gap;
  cv::putText(
    image, line3,
    cv::Point(tx, ty),
    cv::FONT_HERSHEY_SIMPLEX, font_scale,
    cv::Scalar(255, 255, 0), thickness);
}

void drawModeBanner(
  cv::Mat & image,
  uint8_t mode,
  bool tracking,
  bool use_test_mode,
  bool use_local_camera)
{
  std::ostringstream oss;
  oss << "MODE: " << modeToString(mode)
      << " | TRACK: " << (tracking ? "YES" : "NO")
      << " | MODE_SRC: " << (use_test_mode ? "TEST" : "SERIAL")
      << " | IMG_SRC: " << (use_local_camera ? "LOCAL_CAM" : "ROS_TOPIC");

  const std::string text = oss.str();

  int baseline = 0;
  const cv::Size text_size =
    cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.65, 2, &baseline);

  const int x = 10;
  const int y = 30;

  cv::rectangle(
    image,
    cv::Rect(x - 6, y - text_size.height - 8, text_size.width + 12, text_size.height + 12),
    cv::Scalar(0, 0, 0),
    cv::FILLED);

  cv::putText(
    image,
    text,
    cv::Point(x, y),
    cv::FONT_HERSHEY_SIMPLEX,
    0.65,
    cv::Scalar(0, 255, 0),
    2);
}

struct BestTarget
{
  bool found = false;
  Detection det;
  PoseResult pose;
};

BestTarget pickBestTarget(
  const cv::Mat & image,
  Detector * detector,
  const PoseSolver * pose_solver,
  const std::vector<int> & class_mapping,
  const std::vector<int> & pnp_class_ids)
{
  BestTarget best;
  if (detector == nullptr || pose_solver == nullptr) {
    return best;
  }

  const std::vector<Detection> raw_dets = detector->infer(image);
  float best_score = -1.0f;

  for (auto det : raw_dets) {
    det.class_id = remapClassId(det.class_id, class_mapping);
    if (!classAllowed(det.class_id, pnp_class_ids)) {
      continue;
    }

    const PoseResult pose = pose_solver->solve(det);
    if (!pose.success) {
      continue;
    }

    if (det.score > best_score) {
      best_score = det.score;
      best.det = det;
      best.pose = pose;
      best.found = true;
    }
  }

  return best;
}

}  // namespace

class VisionNode : public rclcpp::Node
{
public:
  VisionNode() : Node("smarthome_vision_node")
  {
    declare_parameter<std::string>("image_topic", "/image_raw");
    declare_parameter<std::string>("gimbal_mode_topic", "gimbal_mode");

    declare_parameter<bool>("show_debug", true);
    declare_parameter<std::string>("inference_backend", "openvino");
    declare_parameter<std::string>("openvino_device", "CPU");
    declare_parameter<bool>("use_cuda_preprocess", false);

    declare_parameter<bool>("use_test_mode", false);
    declare_parameter<int>("test_mode", 0);

    declare_parameter<bool>("use_local_camera", true);
    declare_parameter<int>("camera_device_id", 0);
    declare_parameter<int>("camera_width", 640);
    declare_parameter<int>("camera_height", 480);
    declare_parameter<int>("camera_fps", 30);

    declare_parameter<int>("object_input_width", 640);
    declare_parameter<int>("object_input_height", 640);
    declare_parameter<double>("object_conf_threshold", 0.25);
    declare_parameter<double>("object_score_threshold", 0.25);
    declare_parameter<std::string>("object_model_path", "");
    declare_parameter<std::string>("object_engine_path", "");
    declare_parameter<bool>("object_output_keypoints", false);

    declare_parameter<int>("qr_input_width", 640);
    declare_parameter<int>("qr_input_height", 640);
    declare_parameter<double>("qr_conf_threshold", 0.25);
    declare_parameter<double>("qr_score_threshold", 0.25);
    declare_parameter<std::string>("qr_model_path", "");
    declare_parameter<std::string>("qr_engine_path", "");
    declare_parameter<bool>("qr_output_keypoints", true);

    declare_parameter<std::vector<double>>(
      "camera_matrix",
      {800.0, 0.0, 320.0, 0.0, 800.0, 240.0, 0.0, 0.0, 1.0});
    declare_parameter<std::vector<double>>("dist_coeffs", {0.0, 0.0, 0.0, 0.0, 0.0});

    declare_parameter<std::vector<long int>>("class_names", std::vector<long int>{0, 1, 2});
    declare_parameter<std::vector<double>>(
      "class_sizes",
      std::vector<double>{0.083, 0.073, 0.060, 0.050, 0.083, 0.073});

    declare_parameter<std::vector<long int>>("object_model_class_ids", std::vector<long int>{0, 1, 2});
    declare_parameter<std::vector<long int>>("object_pnp_class_ids", std::vector<long int>{0, 1});
    declare_parameter<std::vector<long int>>("qr_model_class_ids", std::vector<long int>{0, 1, 2, 3});

    declare_parameter<double>("qr_size", 0.05);

    show_debug_ = get_parameter("show_debug").as_bool();
    use_local_camera_ = get_parameter("use_local_camera").as_bool();
    camera_device_id_ = get_parameter("camera_device_id").as_int();
    camera_width_ = get_parameter("camera_width").as_int();
    camera_height_ = get_parameter("camera_height").as_int();
    camera_fps_ = get_parameter("camera_fps").as_int();

    auto k = get_parameter("camera_matrix").as_double_array();
    auto d = get_parameter("dist_coeffs").as_double_array();
    auto class_names_ll = get_parameter("class_names").as_integer_array();
    auto class_sizes = get_parameter("class_sizes").as_double_array();
    auto obj_map_ll = get_parameter("object_model_class_ids").as_integer_array();
    auto obj_pnp_ll = get_parameter("object_pnp_class_ids").as_integer_array();
    auto qr_map_ll = get_parameter("qr_model_class_ids").as_integer_array();

    for (const auto v : class_names_ll) {
      class_names_.push_back(static_cast<int>(v));
    }
    for (const auto v : obj_map_ll) {
      object_model_class_ids_.push_back(static_cast<int>(v));
    }
    for (const auto v : obj_pnp_ll) {
      object_pnp_class_ids_.push_back(static_cast<int>(v));
    }
    for (const auto v : qr_map_ll) {
      qr_model_class_ids_.push_back(static_cast<int>(v));
    }

    CameraIntrinsics cam;
    cam.camera_matrix =
      (cv::Mat_<double>(3, 3) << k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8]);
    cam.dist_coeffs = cv::Mat(d).clone().reshape(1, 1);

    object_pose_solver_ = std::make_unique<PoseSolver>();
    object_pose_solver_->set_camera(cam);
    object_pose_solver_->set_class_size_map(buildSizeMap(class_names_, class_sizes));

    const float qr_size = static_cast<float>(get_parameter("qr_size").as_double());
    qr_pose_solver_ = std::make_unique<PoseSolver>();
    qr_pose_solver_->set_camera(cam);
    qr_pose_solver_->set_class_size_map(buildSquareSizeMap(qr_model_class_ids_, qr_size));

    const std::string backend = get_parameter("inference_backend").as_string();
    const std::string openvino_device = get_parameter("openvino_device").as_string();
    std::string object_model_path = get_parameter("object_model_path").as_string();
    std::string qr_model_path = get_parameter("qr_model_path").as_string();

    if (object_model_path.empty() && backend == "tensorrt") {
      object_model_path = get_parameter("object_engine_path").as_string();
    }
    if (qr_model_path.empty() && backend == "tensorrt") {
      qr_model_path = get_parameter("qr_engine_path").as_string();
    }

    object_detector_ = std::make_unique<Detector>(
      backend,
      object_model_path,
      openvino_device,
      get_parameter("object_input_width").as_int(),
      get_parameter("object_input_height").as_int(),
      static_cast<float>(get_parameter("object_conf_threshold").as_double()),
      static_cast<float>(get_parameter("object_score_threshold").as_double()),
      get_parameter("object_output_keypoints").as_bool(),
      get_parameter("use_cuda_preprocess").as_bool());

    qr_detector_ = std::make_unique<Detector>(
      backend,
      qr_model_path,
      openvino_device,
      get_parameter("qr_input_width").as_int(),
      get_parameter("qr_input_height").as_int(),
      static_cast<float>(get_parameter("qr_conf_threshold").as_double()),
      static_cast<float>(get_parameter("qr_score_threshold").as_double()),
      get_parameter("qr_output_keypoints").as_bool(),
      get_parameter("use_cuda_preprocess").as_bool());

    RCLCPP_INFO(
      this->get_logger(),
      "Vision inference backend: %s, device: %s",
      backend.c_str(), openvino_device.c_str());

    gimbal_mode_sub_ = create_subscription<std_msgs::msg::UInt8>(
      get_parameter("gimbal_mode_topic").as_string(), 10,
      [this](const std_msgs::msg::UInt8::SharedPtr msg) {
        gimbal_mode_ = msg->data;
      });

    pub_ = create_publisher<smarthome_vision::msg::DetectedTarget>("detected_target", 10);

    if (!use_local_camera_) {
      sub_ = create_subscription<sensor_msgs::msg::Image>(
        get_parameter("image_topic").as_string(), 10,
        std::bind(&VisionNode::imageCallback, this, _1));
      RCLCPP_INFO(this->get_logger(), "Using ROS image topic input.");
    } else {
      if (!cap_.open(camera_device_id_)) {
        throw std::runtime_error("failed to open local camera device");
      }

      if (camera_width_ > 0) {
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, camera_width_);
      }
      if (camera_height_ > 0) {
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, camera_height_);
      }
      if (camera_fps_ > 0) {
        cap_.set(cv::CAP_PROP_FPS, camera_fps_);
      }

      const int period_ms = camera_fps_ > 0 ? std::max(1, 1000 / camera_fps_) : 33;
      timer_ = create_wall_timer(
        std::chrono::milliseconds(period_ms),
        std::bind(&VisionNode::cameraTimerCallback, this));

      RCLCPP_INFO(
        this->get_logger(),
        "Using local camera device %d, requested %dx%d @ %d FPS.",
        camera_device_id_, camera_width_, camera_height_, camera_fps_);
    }

    if (show_debug_) {
      cv::namedWindow("smarthome_vision_debug", cv::WINDOW_NORMAL);
    }

    RCLCPP_INFO(this->get_logger(), "Vision Node started.");
  }

  ~VisionNode()
  {
    if (cap_.isOpened()) {
      cap_.release();
    }
    if (show_debug_) {
      cv::destroyWindow("smarthome_vision_debug");
    }
  }

private:
  uint8_t getCurrentMode()
  {
    const bool use_test_mode = get_parameter("use_test_mode").as_bool();
    if (use_test_mode) {
      return static_cast<uint8_t>(get_parameter("test_mode").as_int());
    }

    return gimbal_mode_;
  }

  builtin_interfaces::msg::Time nowAsBuiltinTime() const
  {
    return this->now();
  }

  void publishEmptyResult(const builtin_interfaces::msg::Time & stamp, uint8_t mode)
  {
    smarthome_vision::msg::DetectedTarget out;
    out.stamp = stamp;
    out.mode = mode;
    out.tracking = false;
    out.class_id = -1;
    out.score = 0.0f;
    out.x = 0.0f;
    out.y = 0.0f;
    out.z = 0.0f;
    pub_->publish(out);
  }

  void publishFoundResult(
    const builtin_interfaces::msg::Time & stamp,
    uint8_t mode,
    const Detection & det,
    const PoseResult & pose)
  {
    smarthome_vision::msg::DetectedTarget out;
    out.stamp = stamp;
    out.mode = mode;
    out.tracking = true;
    out.class_id = det.class_id;
    out.score = det.score;
    out.x = static_cast<float>(pose.tvec[0]);
    out.y = static_cast<float>(pose.tvec[1]);
    out.z = static_cast<float>(pose.tvec[2]);
    out.corners_uv.resize(8);
    for (int i = 0; i < 4; ++i) {
      out.corners_uv[2 * i] = det.corners[i].x;
      out.corners_uv[2 * i + 1] = det.corners[i].y;
    }
    pub_->publish(out);
  }

  void processFrame(const cv::Mat & image, const builtin_interfaces::msg::Time & stamp)
  {
    const bool use_test_mode = get_parameter("use_test_mode").as_bool();
    const uint8_t mode = getCurrentMode();

    BestTarget best;

    if (mode == static_cast<uint8_t>(VisionMode::IDLE)) {
      publishEmptyResult(stamp, mode);

      if (show_debug_) {
        cv::Mat vis = image.clone();
        drawModeBanner(vis, mode, false, use_test_mode, use_local_camera_);
        cv::imshow("smarthome_vision_debug", vis);
        cv::waitKey(1);
      }
      return;
    }

    if (mode == static_cast<uint8_t>(VisionMode::DETECT_OBJECT)) {
      best = pickBestTarget(
        image,
        object_detector_.get(),
        object_pose_solver_.get(),
        object_model_class_ids_,
        object_pnp_class_ids_);
    } else if (mode == static_cast<uint8_t>(VisionMode::DETECT_QR)) {
      best = pickBestTarget(
        image,
        qr_detector_.get(),
        qr_pose_solver_.get(),
        qr_model_class_ids_,
        {});
    } else {
      publishEmptyResult(stamp, static_cast<uint8_t>(VisionMode::IDLE));
      return;
    }

    if (best.found) {
      publishFoundResult(stamp, mode, best.det, best.pose);
    } else {
      publishEmptyResult(stamp, mode);
    }

    if (show_debug_) {
      cv::Mat vis = image.clone();

      if (best.found) {
        drawDetectionDebug(
          vis,
          best.det,
          mode == static_cast<uint8_t>(VisionMode::DETECT_OBJECT) ? "OBJ" : "QR");
        drawPoseInsideBox(vis, best.det, best.pose);
      }

      drawModeBanner(vis, mode, best.found, use_test_mode, use_local_camera_);

      const int scale = 2;
      cv::Mat vis_big;
      cv::resize(
        vis,
        vis_big,
        cv::Size(vis.cols * scale, vis.rows * scale),
        0,
        0,
        cv::INTER_NEAREST);

      cv::resizeWindow("smarthome_vision_debug", vis_big.cols, vis_big.rows);
      cv::imshow("smarthome_vision_debug", vis_big);
      cv::waitKey(1);
    }
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat image;
    try {
      image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const std::exception & e) {
      RCLCPP_WARN(this->get_logger(), "cv_bridge error: %s", e.what());
      return;
    }

    processFrame(image, msg->header.stamp);
  }

  void cameraTimerCallback()
  {
    if (!cap_.isOpened()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Local camera is not opened.");
      return;
    }

    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(), *this->get_clock(), 2000,
        "Failed to read frame from local camera.");
      return;
    }

    processFrame(frame, nowAsBuiltinTime());
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr gimbal_mode_sub_;
  rclcpp::Publisher<smarthome_vision::msg::DetectedTarget>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::unique_ptr<Detector> object_detector_;
  std::unique_ptr<Detector> qr_detector_;
  std::unique_ptr<PoseSolver> object_pose_solver_;
  std::unique_ptr<PoseSolver> qr_pose_solver_;

  uint8_t gimbal_mode_ = static_cast<uint8_t>(VisionMode::IDLE);

  cv::VideoCapture cap_;

  bool show_debug_ = true;
  bool use_local_camera_ = true;
  int camera_device_id_ = 0;
  int camera_width_ = 640;
  int camera_height_ = 480;
  int camera_fps_ = 30;

  std::vector<int> class_names_;
  std::vector<int> object_model_class_ids_;
  std::vector<int> object_pnp_class_ids_;
  std::vector<int> qr_model_class_ids_;
};

}  // namespace smarthome_vision

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<smarthome_vision::VisionNode>());
  rclcpp::shutdown();
  return 0;
}
