# 采摘巡航决策方案

## 当前导航发点方式

当前 `PubNav2Goal` 发布的是：

```text
/goal_pose
geometry_msgs/msg/PoseStamped
```

Nav2 的 `bt_navigator` 会订阅这个话题并执行导航。这个方式绕过了 `SendNav2Goal` 的 halt bug，但如果在行为树 tick 中反复发布同一个 goal，就不适合再用 action status 判断“刚刚这个目标是否完成”。

如果机器人命名空间是 `red_standard_robot1`，可以监听：

```text
/red_standard_robot1/navigate_to_pose/_action/status
action_msgs/msg/GoalStatusArray
```

成功到点的状态值是 `status = 4`，对应 `action_msgs/msg/GoalStatus.STATUS_SUCCEEDED`。

## 当前实现

比赛主流程放在：

```text
pb2025_sentry_behavior/smart_picking_manager
```

该节点负责完整状态机，默认由 `pb2025_sentry_behavior_launch.py` 启动；`behavior_trees/smarthome.xml` 保留为行为树入口和旧 BT 兼容，不再负责重复发点。

任务管理节点订阅：

| Topic | Type | 用途 |
| --- | --- | --- |
| `/robot_mode` | `std_msgs/UInt8` | 下位机当前模式。 |
| `/detected_target` | `smarthome_vision/DetectedTarget` | 视觉是否看到目标和 xyz。 |
| `/navigate_to_pose/_action/status` | `action_msgs/GoalStatusArray` | 判断导航是否到点。 |

任务管理节点发布：

| Topic | Type | 用途 |
| --- | --- | --- |
| `/goal_pose` | `geometry_msgs/PoseStamped` | 一次发布一个导航点。 |
| `/robot_command` | `std_msgs/UInt8` | 给下位机发命令。 |

## 上位机命令

| 值 | 名称 | 含义 |
| --- | --- | --- |
| 0 | NONE | 无新命令。 |
| 1 | START_SCAN | 到采摘点后，请求下位机开始自动扫描。 |
| 2 | HOLD | 保持。 |
| 3 | RESUME_NAV | 允许继续导航。 |
| 4 | START_DUMP | 到最终点后，请求下位机倾倒果实。 |

## 下位机模式

| 值 | 名称 | 含义 |
| --- | --- | --- |
| 0 | IDLE | 空闲/允许导航。 |
| 1 | AUTO_SCAN | 正在扫描。 |
| 2 | PICKING | 正在采摘。 |
| 3 | PICK_DONE | 采摘完成。 |
| 4 | SCAN_DONE_NO_TARGET | 扫描完成但没有目标。 |
| 5 | DUMPING | 正在倾倒果实。 |
| 6 | DUMP_DONE | 倾倒完成。 |
| 255 | ERROR | 错误。 |

## 状态机

```text
SEND_NAV_GOAL
  发布当前采摘点到 /goal_pose 一次
  -> WAIT_NAV_DONE

WAIT_NAV_DONE
  如果新 goal 的 status=4:
    等待 settle_ms
    发布 START_SCAN
    -> WAIT_SCAN_OR_PICK

WAIT_SCAN_OR_PICK
  如果 detected_target.tracking=true:
    robot_serial_comm 持续把 xyz 发给下位机
  如果 robot_mode=PICK_DONE:
    根据 repeat_after_pick_done 决定重新扫描同一点或进入下一个点
  如果 robot_mode=SCAN_DONE_NO_TARGET:
    如果还有采摘点，进入下一个采摘点
    如果采摘点全部完成，进入最终倾倒点
  如果 robot_mode=ERROR:
    发布 HOLD 并停止

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
    任务结束
  如果 robot_mode=ERROR:
    发布 HOLD 并停止
```

## 底盘停止保护

扫描、采摘、倾倒时，仅靠停止发新目标不一定够，因为 Nav2 controller 可能还在发布旧速度。当前短期方案是在 `robot_serial_comm` 中保护：

```text
robot_mode=AUTO_SCAN/PICKING/DUMPING
```

收到这些模式后，串口 `SP` 帧里的 `vx/vy/wz` 会被强制置零。
