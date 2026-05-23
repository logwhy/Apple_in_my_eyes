# robot_serial_comm

独立串口通信功能包：承接原 `smarthome_vision` 中的 `GimbalBridge` 逻辑，并兼容导航 `cmd_vel`。

## 功能

- 订阅 `cmd_vel`（与 `standard_robot_pp_ros2` 一致），将 `vx/vy/wz` 写入下发数据包
- 订阅 `detected_target`（视觉节点输出），将识别目标写入同一数据包
- 以固定频率（默认 200Hz）向下位机发送 **SP** 协议帧
- 接收下位机 **VS** 模式帧，发布 `gimbal_mode` 供视觉节点读取

## 协议变更

在原有 `VisionToGimbal` 末尾、`crc16` 前增加：

```cpp
struct {
  float vx;
  float vy;
  float wz;
} __attribute__((packed)) speed_vector;
```

## 话题

| 方向 | 话题 | 类型 |
|------|------|------|
| 订阅 | `cmd_vel` | `geometry_msgs/Twist` |
| 订阅 | `detected_target` | `smarthome_vision/DetectedTarget` |
| 发布 | `gimbal_mode` | `std_msgs/UInt8` |
| 发布 | `serial_tx_hex` | `std_msgs/String`（调试） |

## 启动

```bash
# 仅通信节点
ros2 launch robot_serial_comm robot_serial_comm.launch.py

# 视觉 + 通信（vision.launch.py 已包含本节点）
ros2 launch smarthome_vision vision.launch.py
```

## 与导航联调

导航栈 `fake_vel_transform` 最终发布 `/cmd_vel`，本节点订阅后即可驱动底盘，无需 `standard_robot_pp_ros2`。

```
Nav2 → cmd_vel_nav2_result → fake_vel_transform → cmd_vel → robot_serial_comm → 下位机
smarthome_vision → detected_target → robot_serial_comm → 下位机
下位机 → robot_serial_comm → gimbal_mode → smarthome_vision
```

## 参数

见 `config/robot_serial_comm.yaml`：

- `serial_device`：串口设备，默认 `/dev/gimbal`
- `baudrate`：波特率，默认 `115200`
- `send_rate_hz`：发送频率，默认 `200`
