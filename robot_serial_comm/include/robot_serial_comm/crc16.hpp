#pragma once

#include <cstdint>

namespace robot_serial_comm
{

uint16_t crc16_modbus(const uint8_t * data, uint16_t length);

}  // namespace robot_serial_comm
