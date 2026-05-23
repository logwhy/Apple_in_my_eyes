#pragma once

#include <cstdint>

namespace robot_serial_comm
{

enum class VisionMode : uint8_t
{
  IDLE = 0,
  DETECT_OBJECT = 1,
  DETECT_QR = 2
};

// 下位机 -> 上位机
struct __attribute__((packed)) GimbalToVision
{
  uint8_t head[2] = {'V', 'S'};
  uint8_t mode = static_cast<uint8_t>(VisionMode::IDLE);
  uint16_t crc16 = 0;
};

// 上位机 -> 下位机（视觉目标 + 底盘速度）
struct __attribute__((packed)) VisionToGimbal
{
  uint8_t head[2] = {'S', 'P'};
  uint8_t mode = static_cast<uint8_t>(VisionMode::IDLE);
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
