# pb2025_sentry_behavior（简化版）

面向 smarthome / 通用场景的行为树决策，保留：

- **Nav2 发点**：`SendNav2Goal` → `navigate_to_pose` Action
- **视觉监听**：`IsTargetDetected` 读取 `smarthome_vision/DetectedTarget`

已移除 RoboMaster 裁判系统、自瞄、追击等插件的编译依赖（旧行为树 XML 仍保留在目录中供参考，但不再构建对应插件）。

## 行为树

| ID | 说明 |
|----|------|
| `smarthome` | 并行监听视觉 + 导航到 `nav_goal`（默认） |
| `nav_only` | 仅导航 |
| `vision_only` | 仅轮询视觉 |

## 视觉判定规则

`IsTargetDetected` 在以下条件同时满足时返回 SUCCESS：

- blackboard 上有 `detected_target` 消息
- 消息时间戳在 `timeout_ms`（默认 500ms）以内
- `tracking == true`
- `class_id >= 0`
- `score > 0`

视觉发空结果（`tracking=false`, `class_id=-1`）或长时间不发话题，均视为无目标。

## 启动

```bash
# 需先启动 Nav2、smarthome_vision、robot_serial_comm
ros2 launch pb2025_sentry_behavior pb2025_sentry_behavior_launch.py
```

## 参数（`params/sentry_behavior.yaml`）

| 参数 | 说明 |
|------|------|
| `vision_topic` | 视觉话题，默认 `detected_target` |
| `nav_goal` | 导航目标字符串 `x;y;yaw`，默认 `2.0;1.0;0.0` |
| `target_tree` | client 执行的行为树 ID，默认 `smarthome` |

## 与 Nav2 的接口

- Action：`navigate_to_pose`（`nav2_msgs/action/NavigateToPose`）
- 目标坐标系：`map`
- 需 Nav2 栈 lifecycle 为 active，且 `map→base` TF 正常
