#pragma once

#include <cstdint>

namespace robot_serial_comm
{

enum class UpperCommand : uint8_t
{
  NONE = 0,
  START_SCAN = 1,
  HOLD = 2,
  RESUME_NAV = 3,
  START_DUMP = 4
};

enum class LowerMode : uint8_t
{
  IDLE = 0,
  AUTO_SCAN = 1,
  PICKING = 2,
  PICK_DONE = 3,
  SCAN_DONE_NO_TARGET = 4,
  DUMPING = 5,
  DUMP_DONE = 6,
  ERROR = 255
};

// Lower controller -> upper computer.
struct __attribute__((packed)) GimbalToVision
{
  uint8_t head[2] = {'V', 'S'};
  uint8_t robot_mode = static_cast<uint8_t>(LowerMode::IDLE);
  uint16_t crc16 = 0;
};

// Upper computer -> lower controller: command + vision target + chassis speed.
struct __attribute__((packed)) VisionToGimbal
{
  uint8_t head[2] = {'S', 'P'};
  uint8_t command = static_cast<uint8_t>(UpperCommand::NONE);
  uint8_t tracking = 0;
  uint8_t class_id = 0;
  float x = 0.0f;
  float y = 0.0f;
  float z = 0.0f;
  struct
  {
    float vx;
    float vy;
    float wz;
  } __attribute__((packed)) speed_vector;
  uint16_t crc16 = 0;
};

}  // namespace robot_serial_comm
