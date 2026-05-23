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

#ifndef PB2025_SENTRY_BEHAVIOR__PB2025_SENTRY_BEHAVIOR_SERVER_HPP_
#define PB2025_SENTRY_BEHAVIOR__PB2025_SENTRY_BEHAVIOR_SERVER_HPP_

#include <behaviortree_cpp/loggers/bt_cout_logger.h>

#include <memory>
#include <string>
#include <vector>

#include "behaviortree_ros2/tree_execution_server.hpp"
#include "rclcpp/rclcpp.hpp"
#include "pb2025_sentry_behavior/custom_types.hpp"

namespace pb2025_sentry_behavior
{

class SentryBehaviorServer : public BT::TreeExecutionServer
{
public:
  explicit SentryBehaviorServer(const rclcpp::NodeOptions & options);

  bool onGoalReceived(const std::string & tree_name, const std::string & payload) override;

  void onTreeCreated(BT::Tree & tree) override;

  std::optional<BT::NodeStatus> onLoopAfterTick(BT::NodeStatus status) override;

  std::optional<std::string> onTreeExecutionCompleted(
    BT::NodeStatus status, bool was_cancelled) override;

private:
  template <typename T>
  void subscribe(
    const std::string & topic, const std::string & bb_key,
    const rclcpp::QoS & qos = rclcpp::QoS(10));

  std::vector<std::shared_ptr<rclcpp::SubscriptionBase>> subscriptions_;
  std::shared_ptr<BT::StdCoutLogger> logger_cout_;
  uint32_t tick_count_ = 0;
  bool use_cout_logger_ = false;
  std::string vision_topic_;
  std::string nav_goal_;
};

}  // namespace pb2025_sentry_behavior

#endif  // PB2025_SENTRY_BEHAVIOR__PB2025_SENTRY_BEHAVIOR_SERVER_HPP_
