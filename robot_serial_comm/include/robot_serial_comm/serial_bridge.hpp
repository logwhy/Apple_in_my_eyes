#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "robot_serial_comm/protocol.hpp"

namespace robot_serial_comm
{

class SerialBridge
{
public:
  SerialBridge(const std::string & device, int baudrate);
  ~SerialBridge();

  bool isOpened() const;

  void setSpeedVector(float vx, float vy, float wz);
  void setCommand(uint8_t command);
  void setVisionTarget(
    bool tracking, uint8_t class_id, float x, float y, float z);

  bool sendPacket();
  bool updateReceive();
  uint8_t getRobotMode() const;
  int lastSendErrno() const;

  std::string buildPacketHex() const;

private:
  VisionToGimbal buildPacketLocked() const;
  bool openPort();
  void closePort();
  bool parseModePacket();

private:
  int fd_ = -1;
  std::string device_;
  int baudrate_ = 115200;

  mutable std::mutex tx_mutex_;
  std::mutex fd_mutex_;
  VisionToGimbal tx_state_{};

  std::atomic<uint8_t> current_robot_mode_{static_cast<uint8_t>(LowerMode::IDLE)};
  std::atomic<int> last_send_errno_{0};
  std::vector<uint8_t> rx_buffer_;
};

}  // namespace robot_serial_comm
