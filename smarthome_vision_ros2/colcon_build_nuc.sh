#!/usr/bin/env bash
# NUC / 无 CUDA：仅 OpenVINO(CPU)。编译前请确认 CMakeLists.txt 已更新（OpenCV 不在第 19 行写死 cuda*）。
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${WS_ROOT}"
rm -rf build/smarthome_vision install/smarthome_vision

colcon build --packages-select smarthome_vision \
  --cmake-args \
  -DSMARTHOME_VISION_WITH_TENSORRT=OFF \
  -DSMARTHOME_VISION_WITH_OPENVINO=ON
