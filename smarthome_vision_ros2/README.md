# smarthome_vision_ros2

这是按你的要求修改后的版本：

- **统一 TensorRT 10** 推理路径
- **OpenCV CUDA** 预处理
- **双 engine** 结构
  - 关键点 engine
  - bbox engine
- 在 `vision.yaml` 中提供：
  - `enable_bbox_fallback`
  - `force_bbox_only`
- 保留“关键点优先、bbox 兜底”的主逻辑
- 保留 `gimbal` 作为串口通信层，发送 `class_id + x + y + z`

## 目录说明

```text
smarthome_vision_ros2/
├── CMakeLists.txt
├── package.xml
├── README.md
├── msg/
│   └── DetectedTarget.msg
├── config/
│   └── vision.yaml
├── launch/
│   └── vision.launch.py
├── include/smarthome_vision/
│   ├── crc16.hpp
│   ├── detector.hpp
│   ├── gimbal_bridge.hpp
│   ├── pose_solver.hpp
│   ├── protocol.hpp
│   ├── trt_detector.hpp
│   └── types.hpp
└── src/
    ├── crc16.cpp
    ├── detector.cpp
    ├── gimbal_bridge.cpp
    ├── pose_solver.cpp
    ├── trt_detector.cpp
    └── vision_node.cpp
```

## 核心逻辑

```text
/camera/image_raw
      |
      v
vision_node
  |- Detector
      |- keypoint TRT10 engine
      |- bbox TRT10 engine
      |- keypoint valid ? keypoint PnP : bbox fallback
  |- PoseSolver
  |- GimbalBridge
  |- /vision/target
```

## YAML 新参数

`config/vision.yaml` 中新增：

- `keypoint_engine_path`
- `use_keypoint_detector`
- `bbox_engine_path`
- `use_bbox_detector`
- `enable_bbox_fallback`
- `force_bbox_only`
- `use_cuda_preprocess`

## 默认 decode 约定

当前 `src/trt_detector.cpp` 里的 `decode()` 使用的是**通用模板**：

- bbox engine：
  `[cx, cy, w, h, score, class_id]`
- keypoint engine：
  `[cx, cy, w, h, score, class_id, x1, y1, x2, y2, x3, y3, x4, y4]`

如果你的 engine 输出布局不同，只需要改这一处。

## 编译

```bash
cd ~/ws
source /opt/ros/humble/setup.bash
colcon build --packages-select smarthome_vision
source install/setup.bash
ros2 launch smarthome_vision vision.launch.py
```

## 注意

这版工程已经按 **TensorRT 10 + OpenCV CUDA** 改完，但 **engine 输出解码格式仍需要和你实际模型对齐**。如果你给出两个 engine 的输出 shape 和每列定义，可以继续把 `decode()` 改成完全贴合你模型的版本。
