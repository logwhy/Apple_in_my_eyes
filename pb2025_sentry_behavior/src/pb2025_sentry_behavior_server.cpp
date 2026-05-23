// Copyright 2025 Lihan Chen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pb2025_sentry_behavior/pb2025_sentry_behavior_server.hpp"
#include "pb2025_sentry_behavior/custom_types.hpp"

#include <filesystem>
#include <fstream>

#include "behaviortree_cpp/xml_parsing.h"
#include "smarthome_vision/msg/detected_target.hpp"

namespace pb2025_sentry_behavior
{

template <typename T>
void SentryBehaviorServer::subscribe(
  const std::string & topic, const std::string & bb_key, const rclcpp::QoS & qos)
{
  auto sub = node()->create_subscription<T>(
    topic, qos,
    [this, bb_key](const typename T::SharedPtr msg) { globalBlackboard()->set(bb_key, *msg); });
  subscriptions_.push_back(sub);
}

SentryBehaviorServer::SentryBehaviorServer(const rclcpp::NodeOptions & options)
: TreeExecutionServer(options)
{
  node()->declare_parameter("use_cout_logger", false);
  node()->declare_parameter("vision_topic", "detected_target");
  node()->declare_parameter("nav_goal", "0.0;0.0;0.0");

  node()->get_parameter("use_cout_logger", use_cout_logger_);
  node()->get_parameter("vision_topic", vision_topic_);
  node()->get_parameter("nav_goal", nav_goal_);

  auto vision_qos = rclcpp::SensorDataQoS();
  subscribe<smarthome_vision::msg::DetectedTarget>(
    vision_topic_, "vision_detected_target", vision_qos);

  RCLCPP_INFO(
    node()->get_logger(),
    "Subscribed vision topic '%s' -> blackboard vision_detected_target, default nav_goal='%s'",
    vision_topic_.c_str(), nav_goal_.c_str());
}

bool SentryBehaviorServer::onGoalReceived(
  const std::string & tree_name, const std::string & payload)
{
  RCLCPP_INFO(
    node()->get_logger(), "onGoalReceived tree='%s' payload='%s'", tree_name.c_str(),
    payload.c_str());
  return true;
}

void SentryBehaviorServer::onTreeCreated(BT::Tree & tree)
{
  if (use_cout_logger_) {
    logger_cout_ = std::make_shared<BT::StdCoutLogger>(tree);
  }
  tick_count_ = 0;

  tree.rootBlackboard()->set<rclcpp::Node::SharedPtr>("node", this->node());
  tree.rootBlackboard()->set<std::string>("nav_goal", nav_goal_);
}

std::optional<BT::NodeStatus> SentryBehaviorServer::onLoopAfterTick(BT::NodeStatus /*status*/)
{
  ++tick_count_;
  return std::nullopt;
}

std::optional<std::string> SentryBehaviorServer::onTreeExecutionCompleted(
  BT::NodeStatus status, bool was_cancelled)
{
  RCLCPP_INFO(
    node()->get_logger(), "onTreeExecutionCompleted status=%d canceled=%d ticks=%d",
    static_cast<int>(status), was_cancelled, tick_count_);
  logger_cout_.reset();
  return treeName() + " completed with status=" + std::to_string(static_cast<int>(status)) +
         " after " + std::to_string(tick_count_) + " ticks";
}

}  // namespace pb2025_sentry_behavior

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  rclcpp::NodeOptions options;
  auto action_server = std::make_shared<pb2025_sentry_behavior::SentryBehaviorServer>(options);

  RCLCPP_INFO(action_server->node()->get_logger(), "Starting SentryBehaviorServer");

  rclcpp::executors::MultiThreadedExecutor exec(
    rclcpp::ExecutorOptions(), 0, false, std::chrono::milliseconds(250));
  exec.add_node(action_server->node());
  exec.spin();
  exec.remove_node(action_server->node());

  std::string xml_models = BT::writeTreeNodesModelXML(action_server->factory());
  std::ofstream file(std::filesystem::path(ROOT_DIR) / "behavior_trees" / "models.xml");
  file << xml_models;

  rclcpp::shutdown();
}
