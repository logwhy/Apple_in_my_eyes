# 上位机与下位机通信协议说明

本文档给下位机使用。串口帧是二进制协议，不是字符串协议。

## 基本约定

- 默认串口：`115200`，`8N1`，无硬件流控。
- 所有结构体按 `__attribute__((packed))` 紧凑排列。
- 浮点数为 IEEE754 `float`，小端平台可直接按内存字节发送。
- CRC 为 Modbus CRC16，初值 `0xFFFF`，多项式 `0xA001`。
- CRC 计算范围：整帧去掉最后 2 字节 `crc16`。
- 帧头：
  - 上位机到下位机：`'S' 'P'`
  - 下位机到上位机：`'V' 'S'`

## 上位机 -> 下位机：SP 帧

上位机以固定频率发送，默认 `200Hz`。下位机应持续接收最新帧，以最新数据为准。

| 字节偏移 | 类型 | 字段 | 说明 |
| --- | --- | --- | --- |
| 0 | uint8 | head[0] | `'S'`，0x53 |
| 1 | uint8 | head[1] | `'P'`，0x50 |
| 2 | uint8 | command | 上位机命令 |
| 3 | uint8 | tracking | 视觉是否有目标，0/1 |
| 4 | uint8 | class_id | 目标类别 |
| 5 | float | x | 目标相机坐标 x，单位 m |
| 9 | float | y | 目标相机坐标 y，单位 m |
| 13 | float | z | 目标相机坐标 z，单位 m |
| 17 | float | vx | 底盘速度 vx |
| 21 | float | vy | 底盘速度 vy |
| 25 | float | wz | 底盘角速度 wz |
| 29 | uint16 | crc16 | Modbus CRC16，小端 |

总长度：31 字节。

`command` 定义：

| 数值 | 名称 | 下位机建议行为 |
| --- | --- | --- |
| 0 | NONE | 无新命令，保持当前状态。 |
| 1 | START_SCAN | 到达采摘点，上位机请求下位机进入自动扫描模式。 |
| 2 | HOLD | 保持原地/保持机构状态。 |
| 3 | RESUME_NAV | 允许上位机继续导航。 |
| 4 | START_DUMP | 到达最终点，上位机请求下位机执行倾倒果实。 |

`tracking/x/y/z`：

- `tracking=1` 时，`x/y/z` 有效。
- `tracking=0` 时，目标无效，下位机不要使用 `x/y/z` 做采摘结算。
- 坐标系为 OpenCV 相机坐标：`x` 向右，`y` 向下，`z` 向前，单位米。

## 下位机 -> 上位机：VS 帧

下位机需要周期性发送当前模式，建议 `20Hz` 到 `100Hz`。上位机会发布为 ROS 话题 `robot_mode`。

| 字节偏移 | 类型 | 字段 | 说明 |
| --- | --- | --- | --- |
| 0 | uint8 | head[0] | `'V'`，0x56 |
| 1 | uint8 | head[1] | `'S'`，0x53 |
| 2 | uint8 | robot_mode | 下位机当前模式 |
| 3 | uint16 | crc16 | Modbus CRC16，小端 |

总长度：5 字节。

`robot_mode` 定义：

| 数值 | 名称 | 上位机理解 |
| --- | --- | --- |
| 0 | IDLE | 空闲/允许导航。 |
| 1 | AUTO_SCAN | 下位机正在自动上下扫描，上位机应保持机器人不动。 |
| 2 | PICKING | 下位机正在采摘，上位机应保持机器人不动。 |
| 3 | PICK_DONE | 本次采摘完成，上位机可请求再次扫描同一点或继续下一个点。 |
| 4 | SCAN_DONE_NO_TARGET | 扫描完成但无目标，上位机切换到下一个采摘点。 |
| 5 | DUMPING | 下位机正在倾倒果实，上位机应保持机器人不动。 |
| 6 | DUMP_DONE | 倾倒完成，整套采摘任务可以结束。 |
| 255 | ERROR | 下位机错误。 |

## 推荐比赛流程

1. 上位机导航到采摘点。
2. 上位机判断 Nav2 到点后等待 1 秒。
3. 上位机发送 `command=START_SCAN`。
4. 下位机收到后进入 `robot_mode=AUTO_SCAN`，执行固定上下扫描。
5. 上位机视觉持续识别目标，并在 `SP` 帧里实时发送 `tracking/x/y/z`。
6. 下位机在扫描期间如果看到 `tracking=1` 且目标可用，进入 `robot_mode=PICKING`。
7. 下位机用 `x/y/z` 做采摘结算并执行采摘，期间机器人保持不动。
8. 采摘完成后：
   - 若还要继续扫描同一点，发送 `robot_mode=PICK_DONE`，上位机会再次发送 `START_SCAN`。
   - 若扫描结束仍无目标，发送 `robot_mode=SCAN_DONE_NO_TARGET`。
9. 上位机收到 `SCAN_DONE_NO_TARGET` 后导航到下一个采摘点。
10. 所有采摘点执行完后，上位机导航到最终倾倒点。
11. 上位机到最终点后发送 `command=START_DUMP`。
12. 下位机进入 `robot_mode=DUMPING` 并执行倾倒，完成后发送 `robot_mode=DUMP_DONE`。

## 注意事项

- `SP` 帧一直高频发送，不代表每帧都是新命令。下位机应对 `command` 做边沿/状态处理，避免重复触发。
- 建议下位机只在 `command` 从非 `START_SCAN` 变为 `START_SCAN`，或内部状态允许时，才启动一次扫描。
- 建议下位机只在 `command` 从非 `START_DUMP` 变为 `START_DUMP`，且已到最终点时，才启动一次倾倒。
- 扫描、采摘、倾倒期间，下位机应忽略底盘速度或主动保持底盘不动。
- 当前上位机通信节点已经做了保护：收到 `robot_mode=AUTO_SCAN/PICKING/DUMPING` 后，下发 `SP` 帧里的 `vx/vy/wz` 会被强制置零。
- 若 CRC 错误，丢弃该帧并继续找下一帧头。
