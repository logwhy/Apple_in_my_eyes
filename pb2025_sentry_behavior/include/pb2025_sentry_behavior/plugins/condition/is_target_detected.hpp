#pragma once

#include <string>

#include "behaviortree_cpp/condition_node.h"
#include "rclcpp/rclcpp.hpp"
#include "smarthome_vision/msg/detected_target.hpp"

namespace pb2025_sentry_behavior
{

class IsTargetDetectedCondition : public BT::SimpleConditionNode
{
public:
  IsTargetDetectedCondition(const std::string & name, const BT::NodeConfig & config);

  static BT::PortsList providedPorts();

  BT::NodeStatus checkTarget();

private:
  rclcpp::Logger logger_ = rclcpp::get_logger("IsTargetDetected");
};

}  // namespace pb2025_sentry_behavior
