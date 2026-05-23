#include "pb2025_sentry_behavior/plugins/condition/is_target_detected.hpp"

namespace pb2025_sentry_behavior
{

namespace
{

bool isValidTarget(const smarthome_vision::msg::DetectedTarget & msg)
{
  return msg.tracking && msg.class_id >= 0 && msg.score > 0.0f;
}

}  // namespace

IsTargetDetectedCondition::IsTargetDetectedCondition(
  const std::string & name, const BT::NodeConfig & config)
: BT::SimpleConditionNode(name, std::bind(&IsTargetDetectedCondition::checkTarget, this), config)
{
}

BT::NodeStatus IsTargetDetectedCondition::checkTarget()
{
  auto msg = getInput<smarthome_vision::msg::DetectedTarget>("key_port");
  if (!msg) {
    setOutput("target_detected", false);
    return BT::NodeStatus::FAILURE;
  }

  int timeout_ms = 500;
  getInput("timeout_ms", timeout_ms);

  if (timeout_ms > 0) {
    try {
      auto node = config().blackboard->get<rclcpp::Node::SharedPtr>("node");
      const rclcpp::Time stamp(msg->stamp.sec, msg->stamp.nanosec, RCL_ROS_TIME);
      const auto age_ms = (node->now() - stamp).nanoseconds() / 1000000;
      if (age_ms > timeout_ms) {
        setOutput("target_detected", false);
        return BT::NodeStatus::FAILURE;
      }

      const bool detected = isValidTarget(msg.value());
      setOutput("target_detected", detected);
      if (detected) {
        RCLCPP_INFO_THROTTLE(
          logger_, *node->get_clock(), 1000,
          "Vision target detected: class_id=%d score=%.2f pos=(%.2f, %.2f, %.2f)",
          msg->class_id, msg->score, msg->x, msg->y, msg->z);
      }
      return detected ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
    } catch (const std::exception & e) {
      RCLCPP_WARN(logger_, "IsTargetDetected: %s", e.what());
    }
  }

  const bool detected = isValidTarget(msg.value());
  setOutput("target_detected", detected);
  return detected ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
}

BT::PortsList IsTargetDetectedCondition::providedPorts()
{
  return {
    BT::InputPort<smarthome_vision::msg::DetectedTarget>(
      "key_port", "{@vision_detected_target}",
      "DetectedTarget from smarthome_vision on blackboard"),
    BT::InputPort<int>(
      "timeout_ms", 500,
      "Message older than this is treated as no target (ms)"),
    BT::OutputPort<bool>(
      "target_detected", false,
      "Whether a valid vision target is currently detected"),
  };
}

}  // namespace pb2025_sentry_behavior

#include "behaviortree_cpp/bt_factory.h"
BT_REGISTER_NODES(factory)
{
  factory.registerNodeType<pb2025_sentry_behavior::IsTargetDetectedCondition>("IsTargetDetected");
}
