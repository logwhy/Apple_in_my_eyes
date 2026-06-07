#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav2_msgs/action/navigate_to_pose.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "smarthome_vision/msg/detected_target.hpp"
#include "std_msgs/msg/u_int8.hpp"
#include "tf2/LinearMath/Quaternion.h"

namespace pb2025_sentry_behavior
{

namespace
{

constexpr uint8_t kCmdNone = 0;
constexpr uint8_t kCmdStartScan = 1;
constexpr uint8_t kCmdHold = 2;
constexpr uint8_t kCmdResumeNav = 3;
constexpr uint8_t kCmdStartDump = 4;

constexpr uint8_t kModeIdle = 0;
constexpr uint8_t kModeAutoScan = 1;
constexpr uint8_t kModePicking = 2;
constexpr uint8_t kModePickDone = 3;
constexpr uint8_t kModeScanDoneNoTarget = 4;
constexpr uint8_t kModeDumping = 5;
constexpr uint8_t kModeDumpDone = 6;
constexpr uint8_t kModeError = 255;

std::optional<geometry_msgs::msg::PoseStamped> parseGoalString(
  const std::string & goal_str,
  const std::string & frame_id,
  const rclcpp::Time & stamp,
  rclcpp::Logger logger)
{
  std::vector<std::string> parts;
  std::stringstream ss(goal_str);
  std::string item;
  while (std::getline(ss, item, ';')) {
    parts.push_back(item);
  }

  if (parts.size() != 3 && parts.size() != 4) {
    RCLCPP_ERROR(
      logger, "Invalid waypoint '%s'. Expected 'x;y;yaw' or 'x;y;z;yaw'.",
      goal_str.c_str());
    return std::nullopt;
  }

  geometry_msgs::msg::PoseStamped goal;
  goal.header.frame_id = frame_id;
  goal.header.stamp = stamp;

  try {
    goal.pose.position.x = std::stod(parts[0]);
    goal.pose.position.y = std::stod(parts[1]);

    double yaw = 0.0;
    if (parts.size() == 3) {
      goal.pose.position.z = 0.0;
      yaw = std::stod(parts[2]);
    } else {
      goal.pose.position.z = std::stod(parts[2]);
      yaw = std::stod(parts[3]);
    }

    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    goal.pose.orientation.x = q.x();
    goal.pose.orientation.y = q.y();
    goal.pose.orientation.z = q.z();
    goal.pose.orientation.w = q.w();
  } catch (const std::exception & e) {
    RCLCPP_ERROR(logger, "Failed to parse waypoint '%s': %s", goal_str.c_str(), e.what());
    return std::nullopt;
  }

  return goal;
}

}  // namespace

class SmartPickingManager : public rclcpp::Node
{
public:
  using NavigateToPose = nav2_msgs::action::NavigateToPose;
  using GoalHandleNavigateToPose = rclcpp_action::ClientGoalHandle<NavigateToPose>;

  SmartPickingManager()
  : Node("smart_picking_manager")
  {
    // 旧参数保留，不再用于发布 /goal_pose，只是为了兼容原来的 yaml。
    goal_topic_ = declare_parameter<std::string>("goal_topic", "goal_pose");
    nav_status_topic_ =
      declare_parameter<std::string>("nav_status_topic", "navigate_to_pose/_action/status");

    // 新增：直接调用 Nav2 action。
    nav_action_name_ = declare_parameter<std::string>("nav_action_name", "navigate_to_pose");

    robot_command_topic_ = declare_parameter<std::string>("robot_command_topic", "robot_command");
    robot_mode_topic_ = declare_parameter<std::string>("robot_mode_topic", "robot_mode");
    vision_topic_ = declare_parameter<std::string>("vision_topic", "detected_target");
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    settle_ms_ = declare_parameter<int>("settle_ms", 1000);
    dump_settle_ms_ = declare_parameter<int>("dump_settle_ms", 1000);
    command_pulse_ms_ = declare_parameter<int>("command_pulse_ms", 250);
    repeat_after_pick_done_ = declare_parameter<bool>("repeat_after_pick_done", true);
    use_final_goal_ = declare_parameter<bool>("use_final_goal", true);
    final_goal_ = declare_parameter<std::string>("final_goal", "0.0;0.0;0.0");
    nav_goals_ = declare_parameter<std::vector<std::string>>(
      "nav_goals", std::vector<std::string>{"2.0;1.0;0.0"});

    nav_client_ = rclcpp_action::create_client<NavigateToPose>(this, nav_action_name_);

    command_pub_ = create_publisher<std_msgs::msg::UInt8>(robot_command_topic_, 10);

    robot_mode_sub_ = create_subscription<std_msgs::msg::UInt8>(
      robot_mode_topic_, 10,
      [this](const std_msgs::msg::UInt8::SharedPtr msg) {
        previous_robot_mode_ = robot_mode_;
        robot_mode_ = msg->data;
      });

    vision_sub_ = create_subscription<smarthome_vision::msg::DetectedTarget>(
      vision_topic_, rclcpp::SensorDataQoS(),
      [this](const smarthome_vision::msg::DetectedTarget::SharedPtr msg) {
        latest_target_ = *msg;
      });

    timer_ = create_wall_timer(
      std::chrono::milliseconds(50), std::bind(&SmartPickingManager::tick, this));

    RCLCPP_INFO(
      get_logger(),
      "SmartPickingManager started. goals=%zu, action='%s', command='%s', mode='%s', "
      "legacy_goal_topic='%s', legacy_status='%s'",
      nav_goals_.size(), nav_action_name_.c_str(), robot_command_topic_.c_str(),
      robot_mode_topic_.c_str(), goal_topic_.c_str(), nav_status_topic_.c_str());
  }

private:
  enum class State
  {
    SEND_NAV_GOAL,
    WAIT_NAV_DONE,
    WAIT_SETTLE,
    WAIT_SCAN_OR_PICK,
    SEND_FINAL_GOAL,
    WAIT_FINAL_NAV_DONE,
    WAIT_DUMP_SETTLE,
    WAIT_DUMP_DONE,
    FINISHED,
    ERROR_STOP
  };

  void tick()
  {
    updateCommandPulse();

    switch (state_) {
      case State::SEND_NAV_GOAL:
        sendCurrentGoal();
        break;

      case State::WAIT_NAV_DONE:
        if (consumeNavigationSucceeded()) {
          settle_start_ = now();
          state_ = State::WAIT_SETTLE;
          RCLCPP_INFO(
            get_logger(), "Waypoint %zu reached. Waiting %.3f s before scan.",
            current_goal_index_, settle_ms_ / 1000.0);
        }
        break;

      case State::WAIT_SETTLE:
        if ((now() - settle_start_).seconds() >= settle_ms_ / 1000.0) {
          publishCommand(kCmdStartScan);
          state_ = State::WAIT_SCAN_OR_PICK;
          RCLCPP_INFO(get_logger(), "START_SCAN sent at waypoint %zu.", current_goal_index_);
        }
        break;

      case State::WAIT_SCAN_OR_PICK:
        handleLowerMode();
        break;

      case State::SEND_FINAL_GOAL:
        sendFinalGoal();
        break;

      case State::WAIT_FINAL_NAV_DONE:
        if (consumeNavigationSucceeded()) {
          settle_start_ = now();
          state_ = State::WAIT_DUMP_SETTLE;
          RCLCPP_INFO(
            get_logger(), "Final dump point reached. Waiting %.3f s before dump.",
            dump_settle_ms_ / 1000.0);
        }
        break;

      case State::WAIT_DUMP_SETTLE:
        if ((now() - settle_start_).seconds() >= dump_settle_ms_ / 1000.0) {
          publishCommand(kCmdStartDump);
          state_ = State::WAIT_DUMP_DONE;
          RCLCPP_INFO(get_logger(), "START_DUMP sent at final point.");
        }
        break;

      case State::WAIT_DUMP_DONE:
        handleDumpMode();
        break;

      case State::FINISHED:
        publishCommand(kCmdNone);
        break;

      case State::ERROR_STOP:
        publishCommand(kCmdHold);
        break;
    }
  }

  void sendCurrentGoal()
  {
    if (nav_goals_.empty()) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "No nav_goals configured.");
      state_ = State::ERROR_STOP;
      return;
    }

    if (current_goal_index_ >= nav_goals_.size()) {
      RCLCPP_INFO(get_logger(), "All picking waypoints completed.");
      if (use_final_goal_) {
        state_ = State::SEND_FINAL_GOAL;
      } else {
        publishCommand(kCmdNone);
        state_ = State::FINISHED;
      }
      return;
    }

    const auto goal =
      parseGoalString(nav_goals_[current_goal_index_], frame_id_, now(), get_logger());
    if (!goal) {
      state_ = State::ERROR_STOP;
      return;
    }

    if (!sendNavGoal(goal.value(), false)) {
      return;
    }

    state_ = State::WAIT_NAV_DONE;
    RCLCPP_INFO(
      get_logger(), "Sent waypoint %zu/%zu to Nav2 action '%s': %s",
      current_goal_index_ + 1, nav_goals_.size(), nav_action_name_.c_str(),
      nav_goals_[current_goal_index_].c_str());
  }

  void sendFinalGoal()
  {
    const auto goal = parseGoalString(final_goal_, frame_id_, now(), get_logger());
    if (!goal) {
      state_ = State::ERROR_STOP;
      return;
    }

    if (!sendNavGoal(goal.value(), true)) {
      return;
    }

    state_ = State::WAIT_FINAL_NAV_DONE;
    RCLCPP_INFO(
      get_logger(), "Sent final dump point to Nav2 action '%s': %s",
      nav_action_name_.c_str(), final_goal_.c_str());
  }

  bool sendNavGoal(const geometry_msgs::msg::PoseStamped & pose, bool is_final_goal)
  {
    if (!nav_client_->wait_for_action_server(std::chrono::milliseconds(100))) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "Nav2 action server '%s' not available yet. Will retry.",
        nav_action_name_.c_str());
      return false;
    }

    nav_result_available_ = false;
    nav_result_code_ = rclcpp_action::ResultCode::UNKNOWN;

    NavigateToPose::Goal goal_msg;
    goal_msg.pose = pose;

    auto options = rclcpp_action::Client<NavigateToPose>::SendGoalOptions();

    options.goal_response_callback =
      [this, is_final_goal](const GoalHandleNavigateToPose::SharedPtr & goal_handle) {
        if (!goal_handle) {
          RCLCPP_ERROR(
            get_logger(), "%s navigation goal was rejected by Nav2.",
            is_final_goal ? "Final" : "Waypoint");
          nav_result_code_ = rclcpp_action::ResultCode::ABORTED;
          nav_result_available_ = true;
          return;
        }

        RCLCPP_INFO(
          get_logger(), "%s navigation goal accepted by Nav2.",
          is_final_goal ? "Final" : "Waypoint");
      };

    options.feedback_callback =
      [this](
        GoalHandleNavigateToPose::SharedPtr,
        const std::shared_ptr<const NavigateToPose::Feedback> feedback) {
        if (!feedback) {
          return;
        }

        RCLCPP_DEBUG_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "Nav2 feedback: distance_remaining=%.3f",
          feedback->distance_remaining);
      };

    options.result_callback =
      [this, is_final_goal](const GoalHandleNavigateToPose::WrappedResult & result) {
        nav_result_code_ = result.code;
        nav_result_available_ = true;

        switch (result.code) {
          case rclcpp_action::ResultCode::SUCCEEDED:
            RCLCPP_INFO(
              get_logger(), "%s navigation result: SUCCEEDED.",
              is_final_goal ? "Final" : "Waypoint");
            break;

          case rclcpp_action::ResultCode::ABORTED:
            RCLCPP_ERROR(
              get_logger(), "%s navigation result: ABORTED.",
              is_final_goal ? "Final" : "Waypoint");
            break;

          case rclcpp_action::ResultCode::CANCELED:
            RCLCPP_WARN(
              get_logger(), "%s navigation result: CANCELED.",
              is_final_goal ? "Final" : "Waypoint");
            break;

          default:
            RCLCPP_ERROR(
              get_logger(), "%s navigation result: UNKNOWN.",
              is_final_goal ? "Final" : "Waypoint");
            break;
        }
      };

    nav_client_->async_send_goal(goal_msg, options);
    return true;
  }

  bool consumeNavigationSucceeded()
  {
    if (!nav_result_available_) {
      return false;
    }

    const auto code = nav_result_code_;
    nav_result_available_ = false;

    switch (code) {
      case rclcpp_action::ResultCode::SUCCEEDED:
        return true;

      case rclcpp_action::ResultCode::ABORTED:
        RCLCPP_ERROR(get_logger(), "Navigation aborted. Entering ERROR_STOP.");
        state_ = State::ERROR_STOP;
        return false;

      case rclcpp_action::ResultCode::CANCELED:
        RCLCPP_WARN(get_logger(), "Navigation canceled. Entering ERROR_STOP.");
        state_ = State::ERROR_STOP;
        return false;

      default:
        RCLCPP_ERROR(get_logger(), "Navigation failed with unknown result. Entering ERROR_STOP.");
        state_ = State::ERROR_STOP;
        return false;
    }
  }

  void handleLowerMode()
  {
    if (robot_mode_ == kModeError) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Lower controller reported ERROR.");
      state_ = State::ERROR_STOP;
      return;
    }

    if (robot_mode_ == kModePickDone && previous_robot_mode_ != kModePickDone) {
      if (repeat_after_pick_done_) {
        RCLCPP_INFO(get_logger(), "Pick done. Requesting another scan at same waypoint.");
        publishCommand(kCmdStartScan);
      } else {
        advanceGoal();
      }
      return;
    }

    if (
      robot_mode_ == kModeScanDoneNoTarget &&
      previous_robot_mode_ != kModeScanDoneNoTarget)
    {
      RCLCPP_INFO(get_logger(), "Scan done with no target. Moving to next waypoint.");
      advanceGoal();
      return;
    }

    if (latest_target_.tracking) {
      RCLCPP_DEBUG(
        get_logger(), "Vision target: class=%d xyz=(%.3f, %.3f, %.3f)",
        latest_target_.class_id, latest_target_.x, latest_target_.y, latest_target_.z);
    }

    (void)kModeIdle;
    (void)kModeAutoScan;
    (void)kModePicking;
  }

  void handleDumpMode()
  {
    if (robot_mode_ == kModeError) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "Lower controller reported ERROR.");
      state_ = State::ERROR_STOP;
      return;
    }

    if (robot_mode_ == kModeDumpDone && previous_robot_mode_ != kModeDumpDone) {
      RCLCPP_INFO(get_logger(), "Dump done. Smart picking task finished.");
      publishCommand(kCmdNone);
      state_ = State::FINISHED;
      return;
    }

    (void)kModeDumping;
  }

  void advanceGoal()
  {
    publishCommand(kCmdResumeNav);
    current_goal_index_++;
    state_ = State::SEND_NAV_GOAL;
  }

  void publishCommand(uint8_t command)
  {
    std_msgs::msg::UInt8 msg;
    msg.data = command;
    command_pub_->publish(msg);

    if (command != kCmdNone) {
      command_clear_time_ =
        now() + rclcpp::Duration::from_nanoseconds(static_cast<int64_t>(command_pulse_ms_) * 1000000);
      command_clear_pending_ = true;
    }
  }

  void updateCommandPulse()
  {
    if (!command_clear_pending_) {
      return;
    }

    if (now() < command_clear_time_) {
      return;
    }

    std_msgs::msg::UInt8 msg;
    msg.data = kCmdNone;
    command_pub_->publish(msg);
    command_clear_pending_ = false;
  }

  // 兼容旧参数
  std::string goal_topic_;
  std::string nav_status_topic_;

  // 新的 Nav2 action 接口
  std::string nav_action_name_;

  std::string robot_command_topic_;
  std::string robot_mode_topic_;
  std::string vision_topic_;
  std::string frame_id_;
  int settle_ms_ = 1000;
  int dump_settle_ms_ = 1000;
  int command_pulse_ms_ = 250;
  bool repeat_after_pick_done_ = true;
  bool use_final_goal_ = true;
  std::string final_goal_;
  std::vector<std::string> nav_goals_;

  rclcpp_action::Client<NavigateToPose>::SharedPtr nav_client_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr command_pub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr robot_mode_sub_;
  rclcpp::Subscription<smarthome_vision::msg::DetectedTarget>::SharedPtr vision_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  State state_ = State::SEND_NAV_GOAL;
  size_t current_goal_index_ = 0;
  uint8_t robot_mode_ = kModeIdle;
  uint8_t previous_robot_mode_ = kModeIdle;
  rclcpp::Time settle_start_;

  bool nav_result_available_ = false;
  rclcpp_action::ResultCode nav_result_code_ = rclcpp_action::ResultCode::UNKNOWN;

  smarthome_vision::msg::DetectedTarget latest_target_;
  bool command_clear_pending_ = false;
  rclcpp::Time command_clear_time_;
};

}  // namespace pb2025_sentry_behavior

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<pb2025_sentry_behavior::SmartPickingManager>());
  rclcpp::shutdown();
  return 0;
}