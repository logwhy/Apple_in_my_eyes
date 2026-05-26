#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "action_msgs/msg/goal_status.hpp"
#include "action_msgs/msg/goal_status_array.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
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

std::string uuidToString(const std::array<uint8_t, 16> & uuid)
{
  std::ostringstream oss;
  for (const auto byte : uuid) {
    oss << static_cast<int>(byte) << ".";
  }
  return oss.str();
}

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
  SmartPickingManager()
  : Node("smart_picking_manager")
  {
    goal_topic_ = declare_parameter<std::string>("goal_topic", "goal_pose");
    robot_command_topic_ = declare_parameter<std::string>("robot_command_topic", "robot_command");
    robot_mode_topic_ = declare_parameter<std::string>("robot_mode_topic", "robot_mode");
    nav_status_topic_ =
      declare_parameter<std::string>("nav_status_topic", "navigate_to_pose/_action/status");
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

    goal_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(goal_topic_, 10);
    command_pub_ = create_publisher<std_msgs::msg::UInt8>(robot_command_topic_, 10);

    robot_mode_sub_ = create_subscription<std_msgs::msg::UInt8>(
      robot_mode_topic_, 10,
      [this](const std_msgs::msg::UInt8::SharedPtr msg) {
        previous_robot_mode_ = robot_mode_;
        robot_mode_ = msg->data;
      });

    nav_status_sub_ = create_subscription<action_msgs::msg::GoalStatusArray>(
      nav_status_topic_, 10,
      [this](const action_msgs::msg::GoalStatusArray::SharedPtr msg) {
        latest_status_ = *msg;
        have_status_ = true;
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
      "SmartPickingManager started. goals=%zu, status='%s', command='%s', mode='%s'",
      nav_goals_.size(), nav_status_topic_.c_str(), robot_command_topic_.c_str(),
      robot_mode_topic_.c_str());
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
        if (hasNewSucceededGoal()) {
          settle_start_ = now();
          state_ = State::WAIT_SETTLE;
          RCLCPP_INFO(get_logger(), "Waypoint %zu reached. Waiting %.3f s before scan.",
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
        if (hasNewSucceededGoal()) {
          settle_start_ = now();
          state_ = State::WAIT_DUMP_SETTLE;
          RCLCPP_INFO(get_logger(), "Final dump point reached. Waiting %.3f s before dump.",
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

    snapshotSucceededGoalIds();

    const auto goal =
      parseGoalString(nav_goals_[current_goal_index_], frame_id_, now(), get_logger());
    if (!goal) {
      state_ = State::ERROR_STOP;
      return;
    }

    goal_pub_->publish(goal.value());
    state_ = State::WAIT_NAV_DONE;
    RCLCPP_INFO(
      get_logger(), "Published waypoint %zu/%zu: %s", current_goal_index_ + 1,
      nav_goals_.size(), nav_goals_[current_goal_index_].c_str());
  }

  void sendFinalGoal()
  {
    snapshotSucceededGoalIds();

    const auto goal = parseGoalString(final_goal_, frame_id_, now(), get_logger());
    if (!goal) {
      state_ = State::ERROR_STOP;
      return;
    }

    goal_pub_->publish(goal.value());
    state_ = State::WAIT_FINAL_NAV_DONE;
    RCLCPP_INFO(get_logger(), "Published final dump point: %s", final_goal_.c_str());
  }

  void snapshotSucceededGoalIds()
  {
    ignored_succeeded_goal_ids_.clear();
    if (!have_status_) {
      return;
    }

    for (const auto & status : latest_status_.status_list) {
      if (status.status == action_msgs::msg::GoalStatus::STATUS_SUCCEEDED) {
        ignored_succeeded_goal_ids_.insert(uuidToString(status.goal_info.goal_id.uuid));
      }
    }
  }

  bool hasNewSucceededGoal() const
  {
    if (!have_status_) {
      return false;
    }

    for (const auto & status : latest_status_.status_list) {
      if (status.status != action_msgs::msg::GoalStatus::STATUS_SUCCEEDED) {
        continue;
      }

      const auto id = uuidToString(status.goal_info.goal_id.uuid);
      if (ignored_succeeded_goal_ids_.find(id) == ignored_succeeded_goal_ids_.end()) {
        return true;
      }
    }

    return false;
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

  std::string goal_topic_;
  std::string robot_command_topic_;
  std::string robot_mode_topic_;
  std::string nav_status_topic_;
  std::string vision_topic_;
  std::string frame_id_;
  int settle_ms_ = 1000;
  int dump_settle_ms_ = 1000;
  int command_pulse_ms_ = 250;
  bool repeat_after_pick_done_ = true;
  bool use_final_goal_ = true;
  std::string final_goal_;
  std::vector<std::string> nav_goals_;

  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr command_pub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr robot_mode_sub_;
  rclcpp::Subscription<action_msgs::msg::GoalStatusArray>::SharedPtr nav_status_sub_;
  rclcpp::Subscription<smarthome_vision::msg::DetectedTarget>::SharedPtr vision_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  State state_ = State::SEND_NAV_GOAL;
  size_t current_goal_index_ = 0;
  uint8_t robot_mode_ = kModeIdle;
  uint8_t previous_robot_mode_ = kModeIdle;
  rclcpp::Time settle_start_;
  bool have_status_ = false;
  action_msgs::msg::GoalStatusArray latest_status_;
  std::unordered_set<std::string> ignored_succeeded_goal_ids_;
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
