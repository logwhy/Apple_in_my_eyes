# pb2025_sentry_behavior（简化版）

面向 smarthome / 通用场景的行为树决策。

## 发点方式（重要）

| 插件 | 接口 | 行为 | 用途 |
|------|------|------|------|
| **PubNav2Goal**（默认） | 发布 `goal_pose`（PoseStamped） | 发完即 SUCCESS，**不阻塞** | 与 `rmuc_2026` / RViz 一致；`bt_navigator` 订阅该话题并导航 |
| SendNav2Goal（备用） | 调用 `navigate_to_pose` Action | 阻塞直到导航结束 | 仅 `nav_action` 树使用，一般不建议 |

Nav2 的 `bt_navigator` 内部会订阅 `goal_pose`，收到新目标后自行 replan。原版 RM 行为树用的就是 **PubNav2Goal**，不是 Action。

## 行为树

| ID | 说明 |
|----|------|
| `smarthome` | 并行监听视觉 + 循环 `PubNav2Goal`（默认） |
| `nav_only` | 仅循环发 `goal_pose` |
| `nav_action` | 调 Action 发点（阻塞，调试用） |
| `vision_only` | 仅轮询视觉 |

外层 `KeepRunningUntilFailure` + `ForceSuccess` 与原版 RM 树一致：持续 tick、持续发点，决策树不卡在导航过程中。

## 视觉判定

`IsTargetDetected` 在以下条件同时满足时返回 SUCCESS：

- blackboard 上有 `vision_detected_target`
- 消息在 `timeout_ms`（默认 500ms）以内
- `tracking == true`、`class_id >= 0`、`score > 0`

## 启动

```bash
# 需先启动 Nav2（bt_navigator）、smarthome_vision、robot_serial_comm
ros2 launch pb2025_sentry_behavior pb2025_sentry_behavior_launch.py
```

## 参数（`params/sentry_behavior.yaml`）

| 参数 | 说明 |
|------|------|
| `vision_topic` | 视觉话题，默认 `detected_target` |
| `nav_goal` | 导航目标 `x;y;yaw`，默认 `2.0;1.0;0.0` |
| `target_tree` | 行为树 ID，默认 `smarthome` |

## 运行前提

- Nav2 lifecycle active（`bt_navigator` 在跑）
- `map→base` TF 正常
- `cmd_vel` 链路：Nav2 → fake_vel_transform → robot_serial_comm

验证发点：`ros2 topic echo /goal_pose` 应能看到行为树周期发布的 PoseStamped。
