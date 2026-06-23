# smarthome_vision_ros2

## NUC / 无 CUDA 平台编译

报错 `Could NOT find OpenCV (missing: cudaarithm cudaimgproc cudawarping)` 时，**不是业务代码在 NUC 上用了 CUDA**，而是 CMake 打开了 **TensorRT 后端**（会连带要求 OpenCV 的 CUDA 模块）。

```bash
# 清掉旧缓存（很重要，否则 colcon 可能沿用 -DTENSORRT=ON）
rm -rf build/smarthome_vision install/smarthome_vision

colcon build --packages-select smarthome_vision \
  --cmake-args \
  -DSMARTHOME_VISION_WITH_TENSORRT=OFF \
  -DSMARTHOME_VISION_WITH_OPENVINO=ON
```

运行时 `config/vision.yaml` 已默认 `inference_backend: openvino`、`openvino_device: CPU`、`use_cuda_preprocess: false`，与 NUC 匹配。

| 位置 | 是否用 CUDA |
|------|-------------|
| `CMakeLists.txt` 仅当 `SMARTHOME_VISION_WITH_TENSORRT=ON` | 要求 OpenCV cuda* + CUDAToolkit |
| `src/trt_detector.cpp` | TensorRT 推理 + 可选 `cv::cuda::*` 预处理 |
| `src/openvino_detector.cpp` | **不用** CUDA |
| `vision.yaml` `use_cuda_preprocess` | 仅 **tensorrt** 后端生效 |

---

Jetson / 带 TensorRT 的版本：

- **统一 TensorRT 10** 推理路径
- **OpenCV CUDA** 预处理（可选）
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

## 增量修改：xyz offset 与圆心半径结算

本次增量只改视觉节点的目标坐标结算和发布，不改变模型识别输出，也不改变 `detected_target` 消息格式。

### 1. 给下位机发送的 xyz 加人工 offset

视觉节点发布 `detected_target` 前，会给求出的目标坐标加上 `config/vision.yaml` 中的 offset：

```yaml
target_offset_x: 0.0
target_offset_y: 0.0
target_offset_z: 0.0
```

单位都是 m，坐标方向仍然沿用相机坐标系：

- `x`：向右为正。
- `y`：向下为正。
- `z`：向前为正。

例如夹爪实际总是比视觉点偏右 2 cm、偏近 3 cm，可以先试：

```yaml
target_offset_x: 0.02
target_offset_y: 0.0
target_offset_z: -0.03
```

修改后重新编译/启动视觉：

```bash
colcon build --packages-select smarthome_vision
source install/setup.bash
ros2 launch smarthome_vision vision.launch.py
```

调试时看：

```bash
ros2 topic echo /detected_target
```

这里看到的 `x/y/z` 就是已经加过 offset、随后会被 `robot_serial_comm` 打进串口 `SP` 帧的值。

### 2. 物体位姿结算可在圆心半径法和原四点 PnP 间切换

新增参数：

```yaml
object_pose_method: "circle_radius"
circle_radius_scale: 1.0
class_radii: [0.02125, 0.02750, 0.03900]
```

`object_pose_method` 支持两个值：

| 值 | 含义 |
| --- | --- |
| `circle_radius` | 新方法：使用检测框中心点 + 像素半径 + 真实半径估算 xyz。 |
| `bbox_pnp` | 原方法：使用检测框四个角点做 PnP，便于随时切回旧逻辑。 |

`class_radii` 是每个类别真实果子半径，单位 m，顺序必须和 `class_names` 一致。当前示例按 `class_sizes` 的宽高粗略推导，实车建议用尺子量目标果子的平均半径后填入。

如果新方法效果不好，直接改回：

```yaml
object_pose_method: "bbox_pnp"
```

再重新启动视觉节点即可切回原来的四点 PnP。

`circle_radius_scale` 是像素半径修正系数，默认 `1.0`。如果实测发现距离 `z` 整体偏大，说明像素半径估小了，可以把它调大一点；如果 `z` 整体偏小，就调小一点。建议每次只改 `0.03` 到 `0.05`，逐步标定。
