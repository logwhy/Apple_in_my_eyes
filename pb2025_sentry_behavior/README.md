# pb2025_sentry_behavior

面向比赛采摘流程的决策包。

## 当前默认流程

默认启动的是 `smart_picking_manager`，它不用 `SendNav2Goal`，而是发布 `goal_pose` 给 Nav2，并监听 `navigate_to_pose/_action/status` 判断是否到点。

流程：

1. 依次发布 `nav_goals` 中的采摘点。
2. 每个点导航成功后等待 `settle_ms`。
3. 发布 `robot_command=START_SCAN`，等待下位机扫描/采摘。
4. 下位机回 `PICK_DONE` 时可选择继续扫同一点；回 `SCAN_DONE_NO_TARGET` 时进入下一个点。
5. 所有采摘点完成后发布 `final_goal`。
6. 到最终点后发布 `robot_command=START_DUMP`，等待 `DUMP_DONE` 后结束。

## 主要话题

| Topic | Type | 方向 | 用途 |
| --- | --- | --- | --- |
| `goal_pose` | `geometry_msgs/PoseStamped` | publish | 给 Nav2 发导航点。 |
| `navigate_to_pose/_action/status` | `action_msgs/GoalStatusArray` | subscribe | 判断导航是否成功到点，成功值为 `status=4`。 |
| `robot_command` | `std_msgs/UInt8` | publish | 给下位机发 `START_SCAN/START_DUMP` 等命令。 |
| `robot_mode` | `std_msgs/UInt8` | subscribe | 接收下位机 `AUTO_SCAN/PICKING/DUMPING` 等状态。 |
| `detected_target` | `smarthome_vision/DetectedTarget` | subscribe | 调试视觉目标；xyz 实际由通信节点持续转发给下位机。 |

## 参数

见 `params/sentry_behavior.yaml`：

| 参数 | 说明 |
| --- | --- |
| `nav_goals` | 采摘点列表，格式为 `x;y;yaw` 或 `x;y;z;yaw`。 |
| `use_final_goal` | 是否在采摘点完成后去最终倾倒点。 |
| `final_goal` | 最终倾倒点，格式同上。 |
| `settle_ms` | 采摘点到点后等待多久再扫描。 |
| `dump_settle_ms` | 最终点到点后等待多久再倾倒。 |
| `repeat_after_pick_done` | 收到 `PICK_DONE` 后是否在同一点继续扫描。 |

## 行为树 XML

`behavior_trees/smarthome.xml` 现在保留 `smart_picking`/`smarthome` 入口，用于兼容旧行为树启动方式。比赛主逻辑由 `smart_picking_manager` 执行，避免 BT tick 反复发布同一个 goal 导致 Nav2 status 判断不稳定。

## 启动

```bash
ros2 launch pb2025_sentry_behavior pb2025_sentry_behavior_launch.py
```

需要先启动 Nav2、视觉节点、`robot_serial_comm`。
