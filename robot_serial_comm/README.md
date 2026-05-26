# robot_serial_comm

Serial bridge between ROS 2 and the lower controller.

## Topics

| Direction | Topic | Type | Meaning |
| --- | --- | --- | --- |
| subscribe | `cmd_vel` | `geometry_msgs/Twist` | Chassis speed command. |
| subscribe | `detected_target` | `smarthome_vision/DetectedTarget` | Vision target tracking state and xyz. |
| subscribe | `robot_command` | `std_msgs/UInt8` | Upper-computer command to the lower controller. |
| publish | `robot_mode` | `std_msgs/UInt8` | Lower-controller state reported to ROS. |
| publish | `serial_tx_hex` | `std_msgs/String` | Debug hex dump of the outgoing serial frame. |

## Upper -> Lower Frame

Header is `SP`. CRC is Modbus CRC16 over all bytes except the final `crc16` field.

```cpp
struct VisionToGimbal
{
  uint8_t head[2];     // 'S', 'P'
  uint8_t command;     // UpperCommand
  uint8_t tracking;    // 0/1
  uint8_t class_id;    // target class, valid when tracking=1
  float x;             // target x in camera frame, meters
  float y;             // target y in camera frame, meters
  float z;             // target z in camera frame, meters
  float vx;            // chassis vx
  float vy;            // chassis vy
  float wz;            // chassis wz
  uint16_t crc16;
} __attribute__((packed));
```

`command` values:

| Value | Name | Meaning |
| --- | --- | --- |
| 0 | `NONE` | No new command. |
| 1 | `START_SCAN` | Arrived at waypoint; lower controller should start automatic scanning. |
| 2 | `HOLD` | Hold robot/manipulator state. |
| 3 | `RESUME_NAV` | Lower controller allows upper computer to continue navigation. |
| 4 | `START_DUMP` | Arrived at the final point; lower controller should dump collected fruit. |

When the latest received `robot_mode` is `AUTO_SCAN`, `PICKING`, or `DUMPING`, this node forces outgoing
`vx/vy/wz` to zero before calculating CRC.

## Lower -> Upper Frame

Header is `VS`. CRC is Modbus CRC16 over all bytes except the final `crc16` field.

```cpp
struct GimbalToVision
{
  uint8_t head[2];      // 'V', 'S'
  uint8_t robot_mode;   // LowerMode
  uint16_t crc16;
} __attribute__((packed));
```

`robot_mode` values:

| Value | Name | Meaning |
| --- | --- | --- |
| 0 | `IDLE` | Idle or navigation allowed. |
| 1 | `AUTO_SCAN` | Lower controller is scanning. Upper computer should keep the robot still. |
| 2 | `PICKING` | Lower controller is picking. Upper computer should keep the robot still. |
| 3 | `PICK_DONE` | Pick finished. Upper computer can request another scan or continue. |
| 4 | `SCAN_DONE_NO_TARGET` | Scan completed and no target was found. Upper computer should go to the next waypoint. |
| 5 | `DUMPING` | Lower controller is dumping collected fruit. Upper computer should keep the robot still. |
| 6 | `DUMP_DONE` | Dump finished. Full smart-picking task can finish. |
| 255 | `ERROR` | Lower-controller error. |

## Launch

```bash
ros2 launch robot_serial_comm robot_serial_comm.launch.py
```

See `config/robot_serial_comm.yaml` for serial device, baudrate, and topic names.
