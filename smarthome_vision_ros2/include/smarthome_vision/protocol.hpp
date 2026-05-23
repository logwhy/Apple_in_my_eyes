#pragma once

#include <cstdint>

namespace smarthome_vision
{

enum class VisionMode : uint8_t
{
  IDLE = 0,
  DETECT_OBJECT = 1,
  DETECT_QR = 2
};

}  // namespace smarthome_vision
