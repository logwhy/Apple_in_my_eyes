#include "robot_serial_comm/serial_bridge.hpp"

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "robot_serial_comm/crc16.hpp"

namespace robot_serial_comm
{

namespace
{

speed_t toSpeed(int baudrate)
{
  switch (baudrate) {
    case 9600:
      return B9600;
    case 19200:
      return B19200;
    case 38400:
      return B38400;
    case 57600:
      return B57600;
    case 115200:
      return B115200;
    default:
      return B115200;
  }
}

}  // namespace

SerialBridge::SerialBridge(const std::string & device, int baudrate)
: device_(device), baudrate_(baudrate)
{
  openPort();
}

SerialBridge::~SerialBridge()
{
  closePort();
}

bool SerialBridge::openPort()
{
  fd_ = open(device_.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
  if (fd_ < 0) {
    std::cerr << "[SerialBridge] failed to open " << device_ << std::endl;
    return false;
  }

  struct termios tty;
  std::memset(&tty, 0, sizeof(tty));
  if (tcgetattr(fd_, &tty) != 0) {
    std::cerr << "[SerialBridge] tcgetattr failed" << std::endl;
    closePort();
    return false;
  }

  cfsetospeed(&tty, toSpeed(baudrate_));
  cfsetispeed(&tty, toSpeed(baudrate_));

  tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
  tty.c_iflag &= ~IGNBRK;
  tty.c_lflag = 0;
  tty.c_oflag = 0;
  tty.c_cc[VMIN] = 0;
  tty.c_cc[VTIME] = 1;
  tty.c_iflag &= ~(IXON | IXOFF | IXANY);
  tty.c_cflag |= (CLOCAL | CREAD);
  tty.c_cflag &= ~(PARENB | PARODD);
  tty.c_cflag &= ~CSTOPB;
  tty.c_cflag &= ~CRTSCTS;

  if (tcsetattr(fd_, TCSANOW, &tty) != 0) {
    std::cerr << "[SerialBridge] tcsetattr failed" << std::endl;
    closePort();
    return false;
  }

  return true;
}

void SerialBridge::closePort()
{
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

bool SerialBridge::isOpened() const
{
  return fd_ >= 0;
}

void SerialBridge::setSpeedVector(float vx, float vy, float wz)
{
  std::lock_guard<std::mutex> lock(tx_mutex_);
  tx_state_.speed_vector.vx = vx;
  tx_state_.speed_vector.vy = vy;
  tx_state_.speed_vector.wz = wz;
}

void SerialBridge::setVisionTarget(
  uint8_t mode, bool tracking, uint8_t class_id, float x, float y, float z)
{
  std::lock_guard<std::mutex> lock(tx_mutex_);
  tx_state_.mode = mode;
  tx_state_.tracking = tracking ? 1 : 0;
  tx_state_.class_id = class_id;
  tx_state_.x = x;
  tx_state_.y = y;
  tx_state_.z = z;
}

VisionToGimbal SerialBridge::buildPacketLocked() const
{
  VisionToGimbal packet = tx_state_;
  packet.head[0] = 'S';
  packet.head[1] = 'P';
  packet.crc16 =
    crc16_modbus(reinterpret_cast<const uint8_t *>(&packet), sizeof(packet) - 2);
  return packet;
}

std::string SerialBridge::buildPacketHex() const
{
  VisionToGimbal packet;
  {
    std::lock_guard<std::mutex> lock(tx_mutex_);
    packet = buildPacketLocked();
  }

  const auto * bytes = reinterpret_cast<const uint8_t *>(&packet);
  std::ostringstream oss;
  oss << std::uppercase << std::hex << std::setfill('0');

  for (size_t i = 0; i < sizeof(packet); ++i) {
    oss << std::setw(2) << static_cast<int>(bytes[i]);
    if (i + 1 < sizeof(packet)) {
      oss << " ";
    }
  }

  return oss.str();
}

bool SerialBridge::sendPacket()
{
  if (!isOpened()) {
    return false;
  }

  VisionToGimbal packet;
  {
    std::lock_guard<std::mutex> lock(tx_mutex_);
    packet = buildPacketLocked();
  }

  const ssize_t n = write(fd_, &packet, sizeof(packet));
  return n == static_cast<ssize_t>(sizeof(packet));
}

bool SerialBridge::updateReceive()
{
  if (!isOpened()) {
    return false;
  }

  uint8_t temp[64];
  const ssize_t n = read(fd_, temp, sizeof(temp));

  if (n < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return false;
    }
    return false;
  }

  if (n == 0) {
    return false;
  }

  rx_buffer_.insert(rx_buffer_.end(), temp, temp + n);
  return parseModePacket();
}

bool SerialBridge::parseModePacket()
{
  const size_t packet_size = sizeof(GimbalToVision);

  while (rx_buffer_.size() >= packet_size) {
    size_t head_pos = rx_buffer_.size();

    for (size_t i = 0; i + 1 < rx_buffer_.size(); ++i) {
      if (rx_buffer_[i] == 'V' && rx_buffer_[i + 1] == 'S') {
        head_pos = i;
        break;
      }
    }

    if (head_pos == rx_buffer_.size()) {
      rx_buffer_.clear();
      return false;
    }

    if (head_pos > 0) {
      rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + static_cast<long>(head_pos));
    }

    if (rx_buffer_.size() < packet_size) {
      return false;
    }

    GimbalToVision packet;
    std::memcpy(&packet, rx_buffer_.data(), packet_size);

    const uint16_t calc_crc =
      crc16_modbus(reinterpret_cast<uint8_t *>(&packet), packet_size - 2);

    if (calc_crc == packet.crc16) {
      current_mode_ = packet.mode;
      rx_buffer_.erase(rx_buffer_.begin(), rx_buffer_.begin() + static_cast<long>(packet_size));
      return true;
    }

    rx_buffer_.erase(rx_buffer_.begin());
  }

  return false;
}

uint8_t SerialBridge::getMode() const
{
  return current_mode_;
}

}  // namespace robot_serial_comm
