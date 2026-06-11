# 智能采摘方案总说明

本文档描述当前仓库中的比赛方案：上位机负责导航、视觉识别、任务决策和串口打包；下位机负责自动扫描、采摘机构动作、倾倒果实以及向上位机回传当前模式。

当前方案只使用一个视觉模型 `last.onnx` 做目标识别，不再依赖 `qr_detector` 或额外的 `object_detector` 模式切换。

## 总体目标

比赛流程可以概括为：

1. 机器人依次导航到若干采摘点。
2. 每到一个采摘点，上位机等待 1 秒，通知下位机进入自动扫描模式。
3. 下位机执行固定上下扫描。
4. 上位机视觉节点持续识别目标，并把目标 `class_id + x/y/z` 通过串口实时发给下位机。
5. 下位机看到可用目标后进入采摘模式，根据 `x/y/z` 做机构结算并完成采摘。
6. 一个点扫描结束且无目标后，下位机通知上位机进入下一个采摘点。
7. 所有采摘点完成后，机器人导航到最终倾倒点。
8. 到最终点后，上位机通知下位机执行倾倒果实。
9. 下位机倾倒完成后回传完成模式，整套任务结束。

## 软件模块

| 模块 | 路径 | 职责 |
| --- | --- | --- |
| 视觉节点 | `smarthome_vision_ros2` | 运行 `last.onnx`，发布目标识别结果 `detected_target`。 |
| 串口通信 | `robot_serial_comm` | 订阅视觉目标、速度和上位机命令，打包为串口 `SP` 帧；接收下位机 `VS` 帧并发布 `robot_mode`。 |
| 决策节点 | `pb2025_sentry_behavior` | 管理采摘点、最终点、Nav2 到点判断、给下位机发扫描/倾倒命令。 |
| 导航 | `pb2025_sentry_nav` | Nav2 配置、地图导航、到点状态反馈。 |
| 下位机通信参考 | `usb_task.c` / `usb_task.h` | 下位机侧 USB/串口协议解析和回传参考实现。 |

## ROS 话题关系

```text
last.onnx
  -> smarthome_vision_ros2
  -> /detected_target
  -> robot_serial_comm
  -> SP 串口帧：tracking/class_id/x/y/z/vx/vy/wz/command
  -> 下位机

下位机
  -> VS 串口帧：robot_mode
  -> robot_serial_comm
  -> /robot_mode
  -> smart_picking_manager

smart_picking_manager
  -> /goal_pose
  -> Nav2 bt_navigator

Nav2
  -> /navigate_to_pose/_action/status
  -> smart_picking_manager

smart_picking_manager
  -> /robot_command
  -> robot_serial_comm
  -> SP 串口帧 command 字段
```

主要话题：

| 话题 | 类型 | 发布者 | 订阅者 | 说明 |
| --- | --- | --- | --- | --- |
| `detected_target` | `smarthome_vision/DetectedTarget` | 视觉节点 | 串口通信、决策节点 | 目标是否存在、类别、置信度、xyz。 |
| `cmd_vel` | `geometry_msgs/Twist` | Nav2 / 速度链路 | 串口通信 | 底盘速度。 |
| `robot_command` | `std_msgs/UInt8` | 决策节点 | 串口通信 | 上位机给下位机的离散命令。 |
| `robot_mode` | `std_msgs/UInt8` | 串口通信 | 决策节点 | 下位机当前状态。 |
| `goal_pose` | `geometry_msgs/PoseStamped` | 决策节点 | Nav2 | 导航目标点。 |
| `navigate_to_pose/_action/status` | `action_msgs/GoalStatusArray` | Nav2 | 决策节点 | Nav2 action 状态，到点成功为 `status=4`。 |

如果机器人使用命名空间，例如 `red_standard_robot1`，Nav2 状态话题通常会变成：

```text
/red_standard_robot1/navigate_to_pose/_action/status
```

对应地需要修改 `smart_picking_manager.nav_status_topic` 参数。

## 视觉方案

视觉包路径：

```text
smarthome_vision_ros2
```

当前视觉节点的思路：

- 只使用一个 `last.onnx` 模型。
- 默认不再依赖下位机 mode 来切换 QR / object 模式。
- 检测置信度默认设置为 `0.75`。
- 支持多个目标同时出现。
- 多目标选择策略：
  1. 优先选择指定类别，当前默认 `target_priority_class_id=0`，即优先红色。
  2. 同类别下优先选择 `z` 更小的目标，即距离更近。
  3. 再比较横向偏移 `sqrt(x*x + y*y)`，优先选择更居中的目标。
  4. 最后比较置信度。

视觉节点发布：

```text
detected_target
```

主要字段含义：

| 字段 | 含义 |
| --- | --- |
| `tracking` | 是否检测到可用目标。 |
| `class_id` | 目标类别。 |
| `score` | 模型置信度。 |
| `x` | 目标在相机坐标系下的 x，向右为正，单位 m。 |
| `y` | 目标在相机坐标系下的 y，向下为正，单位 m。 |
| `z` | 目标在相机坐标系下的 z，向前为正，单位 m。 |

Windows 下测试模型的脚本：

```text
smarthome_vision_ros2/test_last_onnx.py
```

这个脚本用于直接测试 `last.onnx` 对单张图片的识别效果，会在图片上绘制检测框、识别耗时和 xyz。

## 串口通信协议

通信包路径：

```text
robot_serial_comm
```

完整中文协议文档：

```text
robot_serial_comm/LOWER_CONTROLLER_PROTOCOL.zh.md
```

### 上位机到下位机：SP 帧

帧头：

```text
'S' 'P'
```

字段：

| 字段 | 含义 |
| --- | --- |
| `command` | 上位机命令。 |
| `tracking` | 当前视觉是否有目标。 |
| `class_id` | 当前目标类别。 |
| `x/y/z` | 当前目标相机坐标。 |
| `vx/vy/wz` | 底盘速度。 |
| `crc16` | Modbus CRC16。 |

上位机命令：

| 值 | 名称 | 含义 |
| --- | --- | --- |
| 0 | `NONE` | 无新命令。 |
| 1 | `START_SCAN` | 到达采摘点，请求下位机开始自动扫描。 |
| 2 | `HOLD` | 保持当前状态。 |
| 3 | `RESUME_NAV` | 允许继续导航。 |
| 4 | `START_DUMP` | 到达最终点，请求下位机倾倒果实。 |

### 下位机到上位机：VS 帧

帧头：

```text
'V' 'S'
```

字段：

| 字段 | 含义 |
| --- | --- |
| `robot_mode` | 下位机当前模式。 |
| `crc16` | Modbus CRC16。 |

下位机模式：

| 值 | 名称 | 上位机处理 |
| --- | --- | --- |
| 0 | `IDLE` | 空闲，允许导航。 |
| 1 | `AUTO_SCAN` | 下位机正在扫描，上位机保持底盘不动。 |
| 2 | `PICKING` | 下位机正在采摘，上位机保持底盘不动。 |
| 3 | `PICK_DONE` | 本次采摘完成，上位机可继续请求扫描同一点。 |
| 4 | `SCAN_DONE_NO_TARGET` | 当前点扫描结束且无目标，上位机进入下一个采摘点。 |
| 5 | `DUMPING` | 下位机正在倾倒，上位机保持底盘不动。 |
| 6 | `DUMP_DONE` | 倾倒完成，任务结束。 |
| 255 | `ERROR` | 下位机错误，上位机发送 `HOLD` 并停止任务。 |

### 速度保护

扫描、采摘、倾倒期间，即使 Nav2 仍然输出速度，`robot_serial_comm` 也会在串口打包前强制清零：

```text
robot_mode=AUTO_SCAN/PICKING/DUMPING
```

对应 `SP` 帧中的：

```text
vx = 0
vy = 0
wz = 0
```

这是一层上位机保护。下位机在这些模式下也建议主动忽略底盘速度。

## 决策状态机

决策节点：

```text
pb2025_sentry_behavior/src/smart_picking_manager.cpp
```

默认由 launch 启动：

```text
pb2025_sentry_behavior/launch/pb2025_sentry_behavior_launch.py
```

核心状态机：

```text
SEND_NAV_GOAL
  发布当前采摘点到 /goal_pose 一次
  -> WAIT_NAV_DONE

WAIT_NAV_DONE
  监听 navigate_to_pose/_action/status
  如果新 goal 的 status=4:
    记录到点时间
    -> WAIT_SETTLE

WAIT_SETTLE
  等待 settle_ms
  发布 START_SCAN
  -> WAIT_SCAN_OR_PICK

WAIT_SCAN_OR_PICK
  如果 robot_mode=PICK_DONE:
    如果 repeat_after_pick_done=true:
      再次发布 START_SCAN，继续扫同一点
    否则:
      进入下一个采摘点

  如果 robot_mode=SCAN_DONE_NO_TARGET:
    进入下一个采摘点

  如果所有采摘点已经完成:
    -> SEND_FINAL_GOAL

SEND_FINAL_GOAL
  发布 final_goal 到 /goal_pose 一次
  -> WAIT_FINAL_NAV_DONE

WAIT_FINAL_NAV_DONE
  如果新 goal 的 status=4:
    等待 dump_settle_ms
    发布 START_DUMP
    -> WAIT_DUMP_DONE

WAIT_DUMP_DONE
  如果 robot_mode=DUMP_DONE:
    -> FINISHED

ERROR_STOP
  发布 HOLD
```

注意：视觉 xyz 不需要决策节点转发。视觉节点发布 `detected_target`，通信节点会持续把其中的 `tracking/class_id/x/y/z` 打进串口 `SP` 帧。

## 参数配置

主要参数文件：

```text
pb2025_sentry_behavior/params/sentry_behavior.yaml
```

示例：

```yaml
smart_picking_manager:
  ros__parameters:
    goal_topic: "goal_pose"
    robot_command_topic: "robot_command"
    robot_mode_topic: "robot_mode"
    nav_status_topic: "navigate_to_pose/_action/status"
    vision_topic: "detected_target"
    frame_id: "map"
    settle_ms: 1000
    dump_settle_ms: 1000
    command_pulse_ms: 250
    repeat_after_pick_done: true
    use_final_goal: true
    final_goal: "0.0;0.0;0.0"
    nav_goals:
      - "2.0;1.0;0.0"
      - "3.0;1.5;1.57"
      - "4.0;2.0;3.14"
```

点位格式支持两种：

```text
x;y;yaw
x;y;z;yaw
```

其中 `yaw` 单位是弧度。

常用角度：

| 朝向 | yaw |
| --- | --- |
| 0 度 | `0.0` |
| 90 度 | `1.57` |
| 180 度 | `3.14` |
| -90 度 | `-1.57` |

## 行为树说明

行为树文件：

```text
pb2025_sentry_behavior/behavior_trees/smarthome.xml
```

当前 XML 中的 `smart_picking` 和 `smarthome` 主要作为行为树入口和兼容旧启动方式。完整比赛逻辑放在 `smart_picking_manager` 中，而不是强行写在 XML 中，原因是：

- 当前开源 `SendNav2Goal` 在 halt 时存在 bug，不适合作为主流程 action 节点。
- `PubNav2Goal` 是非阻塞发布，如果放在 BT tick 里循环发布，会不断刷新 goal，影响 Nav2 status 判断。
- 当前流程需要同时处理 Nav2 status、下位机 mode、命令脉冲和最终点倾倒，用 C++ 状态机更稳定。

## 导航朝向控制

Nav2 的 goal 本身支持朝向控制，因为 `goal_pose` 是 `PoseStamped`，里面有 `orientation`。当前 `PubNav2Goal` 和 `smart_picking_manager` 都支持在点位字符串里写 `yaw`。

例如：

```text
"2.0;1.0;1.57"
```

表示导航到 `x=2.0, y=1.0`，最终朝向约 90 度。

但是当前导航参数里：

```text
yaw_goal_tolerance: 6.28
```

这表示最终朝向几乎不检查，基本等于“只要位置到了，朝向随便”。如果希望车严格转到目标朝向，需要把对应 Nav2 参数文件中的 `yaw_goal_tolerance` 改小，例如：

```yaml
yaw_goal_tolerance: 0.15
```

相关文件：

```text
pb2025_sentry_nav/pb2025_nav_bringup/config/reality/nav2_params.yaml
pb2025_sentry_nav/pb2025_nav_bringup/config/reality/nav2_params0.yaml
pb2025_sentry_nav/pb2025_nav_bringup/config/simulation/nav2_params.yaml
```

实车优先改 `reality/nav2_params.yaml`。如果是全向底盘，还要注意 controller 是否允许最终原地旋转到目标 yaw。

## 推荐启动顺序

`standard_robot_pp_ros2` 只是参考包，不参与本项目启动。当前识果机器人项目主要启动四部分：导航、视觉、通信、决策。

每个终端都先 source 工作空间：

```bash
cd ~/ros_ws
source install/setup.bash
```

### 1. 启动导航

实车导航：

```bash
ros2 launch pb2025_nav_bringup rm_navigation_reality_launch.py \
  use_robot_state_pub:=True \
  use_rviz:=True
```

如果你已经由别的节点发布了机器人 TF，可以把 `use_robot_state_pub` 改成 `False`。当前项目不启动 `standard_robot_pp_ros2` 时，通常先用 `True`。

### 2. 启动视觉和通信

```bash
ros2 launch smarthome_vision vision.launch.py
```

注意：这个 launch 已经包含了 `robot_serial_comm.launch.py`，会同时启动串口通信节点，所以不要再重复启动 `robot_serial_comm`。

如果只想单独调通信，才使用：

```bash
ros2 launch robot_serial_comm robot_serial_comm.launch.py
```

### 3. 启动决策

```bash
ros2 launch pb2025_sentry_behavior pb2025_sentry_behavior_launch.py
```

默认 launch 中：

- `enable_smart_picking=true`
- `enable_legacy_bt=false`

也就是说默认启动 `smart_picking_manager`，不启动旧 BT server/client。

## 调试建议

### 检查视觉输出

```bash
ros2 topic echo /detected_target
```

重点看：

```text
tracking
class_id
score
x
y
z
```

### 检查下位机模式

```bash
ros2 topic echo /robot_mode
```

正常流程中应该能看到：

```text
0 -> 1 -> 2 -> 3 -> 1 -> 4 -> ...
```

最终倾倒时：

```text
5 -> 6
```

### 检查上位机命令

```bash
ros2 topic echo /robot_command
```

到采摘点后应出现：

```text
1
```

到最终倾倒点后应出现：

```text
4
```

### 检查导航到点状态

```bash
ros2 topic echo /navigate_to_pose/_action/status
```

如果带命名空间：

```bash
ros2 topic echo /red_standard_robot1/navigate_to_pose/_action/status
```

成功到点的状态是：

```text
status: 4
```

### 检查串口发送内容

```bash
ros2 topic echo /serial_tx_hex
```

可以看到上位机发给下位机的 `SP` 帧十六进制内容，用于确认 `command`、`tracking`、`xyz` 是否正在变化。

## 下位机需要实现的逻辑

下位机至少需要支持：

1. 持续解析 `SP` 帧。
2. 对 `command` 做边沿触发，避免高频重复触发。
3. 收到 `START_SCAN` 后进入 `AUTO_SCAN`。
4. 扫描中如果 `tracking=1` 且目标可用，进入 `PICKING`。
5. 采摘完成后发送 `PICK_DONE`。
6. 当前点扫描结束且无目标后发送 `SCAN_DONE_NO_TARGET`。
7. 收到 `START_DUMP` 后进入 `DUMPING`。
8. 倾倒完成后发送 `DUMP_DONE`。
9. 异常时发送 `ERROR`。
10. 周期性发送 `VS` 帧，上位机依靠它判断当前任务状态。

仓库根目录的 `usb_task.c` / `usb_task.h` 已经给出一份通信侧参考实现。

## 当前方案的关键约定

- 视觉只负责识别，不根据下位机 mode 切换模型。
- 决策只发离散命令，不直接控制采摘机构。
- 串口通信节点负责把视觉 xyz 和命令合并成一帧发给下位机。
- 下位机负责扫描、采摘、倾倒的具体动作。
- 上位机在扫描、采摘、倾倒时会尽量保证底盘速度为 0。
- 所有采摘点完成后，必须去最终点执行倾倒，除非 `use_final_goal=false`。

## Jetson Orin NX CUDA 导航加速入口

`cuda` 分支在 `pb2025_sentry_nav` 中新增了 `pb_cuda_pointcloud` 公共点云加速包，面向 Jetson Orin NX / CUDA 12.6 / `sm_87`。本次只加速点云预处理、点云变换、地形体素下采样和 SLAM 点云转 LaserScan 等 CPU 热点，不修改导航速度、RViz 显示、地图计算频率、Nav2 controller 或 costmap 参数。

最新代码审查已补齐 PCL/CUDA 构建依赖、实际 CUDA 体素/LaserScan kernel，以及 CUDA 失败时的原 CPU/PCL 回退说明。

针对 NX CPU 过载导致点云拖漂的场景，`point_lio` 还新增了内部输入队列防积压保护，过载时保留最新点云而不是一直追旧帧；详细参数仍在导航 README。

详细构建、开关和真机检查方式见：

```text
pb2025_sentry_nav/README.md
```
