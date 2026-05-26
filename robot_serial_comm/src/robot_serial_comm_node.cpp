#include <chrono>
#include <algorithm>
#include <memory>
#include <string>

#include "geometry_msgs/msg/twist.hpp"
#include "rclcpp/rclcpp.hpp"
#include "smarthome_vision/msg/detected_target.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/u_int8.hpp"

#include "robot_serial_comm/serial_bridge.hpp"

namespace robot_serial_comm
{

class RobotSerialCommNode : public rclcpp::Node
{
public:
  RobotSerialCommNode()
  : Node("robot_serial_comm")
  {
    const std::string device = declare_parameter<std::string>("serial_device", "/dev/gimbal");
    const int baudrate = declare_parameter<int>("baudrate", 115200);
    const int send_rate_hz = declare_parameter<int>("send_rate_hz", 200);
    const std::string cmd_vel_topic = declare_parameter<std::string>("cmd_vel_topic", "cmd_vel");
    const std::string vision_topic =
      declare_parameter<std::string>("vision_topic", "detected_target");
    const std::string robot_command_topic =
      declare_parameter<std::string>("robot_command_topic", "robot_command");
    const std::string robot_mode_topic =
      declare_parameter<std::string>("robot_mode_topic", "robot_mode");
    const std::string serial_tx_hex_topic =
      declare_parameter<std::string>("serial_tx_hex_topic", "serial_tx_hex");

    bridge_ = std::make_unique<SerialBridge>(device, baudrate);
    if (!bridge_->isOpened()) {
      RCLCPP_ERROR(get_logger(), "Failed to open serial device: %s", device.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "Serial opened: %s @ %d", device.c_str(), baudrate);
    }

    cmd_vel_sub_ = create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic, rclcpp::QoS(10),
      [this](const geometry_msgs::msg::Twist::SharedPtr msg) {
        bridge_->setSpeedVector(
          static_cast<float>(msg->linear.x),
          static_cast<float>(msg->linear.y),
          static_cast<float>(msg->angular.z));
      });

    vision_sub_ = create_subscription<smarthome_vision::msg::DetectedTarget>(
      vision_topic, rclcpp::QoS(10),
      [this](const smarthome_vision::msg::DetectedTarget::SharedPtr msg) {
        const uint8_t class_id =
          msg->class_id < 0 ? 0 : static_cast<uint8_t>(msg->class_id);
        bridge_->setVisionTarget(
          msg->tracking, class_id, msg->x, msg->y, msg->z);
      });

    robot_command_sub_ = create_subscription<std_msgs::msg::UInt8>(
      robot_command_topic, rclcpp::QoS(10),
      [this](const std_msgs::msg::UInt8::SharedPtr msg) {
        bridge_->setCommand(msg->data);
      });

    robot_mode_pub_ = create_publisher<std_msgs::msg::UInt8>(robot_mode_topic, 10);
    serial_tx_hex_pub_ = create_publisher<std_msgs::msg::String>(serial_tx_hex_topic, 10);

    const auto send_period = std::chrono::milliseconds(
      std::max(1, 1000 / std::max(1, send_rate_hz)));

    send_timer_ = create_wall_timer(send_period, [this]() { onSendTimer(); });

    const auto rx_period = std::chrono::milliseconds(5);
    rx_timer_ = create_wall_timer(rx_period, [this]() { onReceiveTimer(); });

    RCLCPP_INFO(
      get_logger(),
      "Subscribed cmd_vel: %s, vision: %s, command: %s, publishing mode: %s",
      cmd_vel_topic.c_str(), vision_topic.c_str(), robot_command_topic.c_str(),
      robot_mode_topic.c_str());
  }

private:
  void onSendTimer()
  {
    if (!bridge_->isOpened()) {
      return;
    }

    if (!bridge_->sendPacket()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Serial send failed");
      return;
    }

    std_msgs::msg::String hex_msg;
    hex_msg.data = bridge_->buildPacketHex();
    serial_tx_hex_pub_->publish(hex_msg);
  }

  void onReceiveTimer()
  {
    if (!bridge_->isOpened()) {
      return;
    }

    if (bridge_->updateReceive()) {
      std_msgs::msg::UInt8 mode_msg;
      mode_msg.data = bridge_->getRobotMode();
      robot_mode_pub_->publish(mode_msg);
    }
  }

  std::unique_ptr<SerialBridge> bridge_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;
  rclcpp::Subscription<smarthome_vision::msg::DetectedTarget>::SharedPtr vision_sub_;
  rclcpp::Subscription<std_msgs::msg::UInt8>::SharedPtr robot_command_sub_;
  rclcpp::Publisher<std_msgs::msg::UInt8>::SharedPtr robot_mode_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr serial_tx_hex_pub_;
  rclcpp::TimerBase::SharedPtr send_timer_;
  rclcpp::TimerBase::SharedPtr rx_timer_;
};

}  // namespace robot_serial_comm

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<robot_serial_comm::RobotSerialCommNode>());
  rclcpp::shutdown();
  return 0;
}
