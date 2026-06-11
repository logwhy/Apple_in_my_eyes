#!/usr/bin/env python3
"""Detect objects in images using ONNX or OpenVINO.

Dependencies:
  conda activate rs
  pip install onnxruntime openvino opencv-python numpy

Examples:
  python test_last_onnx.py path/to/images --show
  python test_last_onnx.py image.jpg --conf 0.35
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    import cv2
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}\n"
        "Install it with: pip install opencv-python numpy"
    ) from exc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_PATH = r"C:\Users\38480\Desktop\database\apple_detection\weights\last.onnx"
OPENVINO_MODEL_PATH = SCRIPT_DIR / "assets" / "last_openvino_model" / "last.xml"

IMG_SIZE = 640
CONF_THRESHOLD = 0.75
IOU_THRESHOLD = 0.45
DECODE_MODE = "auto"
PREPROCESS_MODE = "resize"
CLASS_NAMES = ["0", "1", "2"]
SHOW_RESULT = True
BACKEND = "both"
OPENVINO_DEVICE = "CPU"
WARMUP_RUNS = 1
REPEAT_RUNS = 1

# Copied from config/vision.yaml.
CAMERA_MATRIX = np.array(
    [
        [816.011939, 0.000000, 305.516996],
        [0.000000, 812.516912, 226.442214],
        [0.000000, 0.000000, 1.000000],
    ],
    dtype=np.float64,
)
DIST_COEFFS = np.array([-0.399955, 0.075996, 0.005002, 0.002332, 0.000000], dtype=np.float64)

CLASS_SIZES = {
    0: (0.083, 0.073),
    1: (0.060, 0.050),
    2: (0.083, 0.073),
}


@dataclass
class PreprocessMeta:
    image_w: int
    image_h: int
    input_w: int
    input_h: int
    mode: str
    scale_x: float
    scale_y: float
    ratio: float = 1.0
    pad_x: float = 0.0
    pad_y: float = 0.0


@dataclass
class Detection:
    class_id: int
    score: float
    box: np.ndarray
    xyz: np.ndarray | None = None
    reproj_error: float | None = None


class Runner:
    """Base class for inference runners."""
    def __init__(self, name: str, input_w: int, input_h: int):
        self.name = name
        self.input_w = input_w
        self.input_h = input_h
    
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ONNXRunner(Runner):
    def __init__(self, model_path: Path, fallback_size: int):
        super().__init__("ONNXRuntime CPU", 0, 0)
        if not model_path.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
        
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit(
                "onnxruntime is required for --backend onnx/both.\n"
                "Install it with: pip install onnxruntime"
            ) from exc
        
        providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_meta = self.session.get_inputs()[0]
        shape = self.input_meta.shape
        if len(shape) != 4:
            raise ValueError(f"Expected NCHW 4D input, got shape: {shape}")
        
        self.input_h = static_dim(shape[2], fallback_size)
        self.input_w = static_dim(shape[3], fallback_size)
        self.input_name = self.input_meta.name
    
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: tensor})[0]


class OpenVINORunner(Runner):
    def __init__(self, model_path: Path, device: str, fallback_size: int):
        super().__init__(f"OpenVINO {device}", 0, 0)
        if not model_path.is_file():
            raise FileNotFoundError(f"OpenVINO model does not exist: {model_path}")
        
        try:
            import openvino as ov
        except ImportError as exc:
            raise SystemExit(
                "openvino is required for --backend openvino/both.\n"
                "Install it with: pip install openvino"
            ) from exc
        
        core = ov.Core()
        model = core.read_model(str(model_path))
        input_port = model.input(0)
        if input_port.partial_shape.is_dynamic:
            model.reshape({input_port.get_any_name(): [1, 3, fallback_size, fallback_size]})
            input_port = model.input(0)
        
        shape = list(input_port.shape)
        if len(shape) != 4:
            raise ValueError(f"Expected OpenVINO NCHW 4D input, got shape: {shape}")
        
        self.input_h = int(shape[2])
        self.input_w = int(shape[3])
        
        compiled = core.compile_model(
            model,
            device,
            {"PERFORMANCE_HINT": "LATENCY"},
        )
        self.compiled = compiled
        self.input_name = compiled.input(0).get_any_name()
        self.output_port = compiled.output(0)
    
    def infer(self, tensor: np.ndarray) -> np.ndarray:
        outputs = self.compiled({self.input_name: tensor})
        return np.asarray(outputs[self.output_port])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect objects in images using ONNX or OpenVINO."
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Image file or image directory to process.",
    )
    parser.add_argument(
        "--backend",
        choices=("onnx", "openvino", "both"),
        default=BACKEND,
        help="Inference backend to use.",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to ONNX model.")
    parser.add_argument(
        "--openvino-model",
        type=Path,
        default=OPENVINO_MODEL_PATH,
        help="Path to OpenVINO .xml IR model.",
    )
    parser.add_argument(
        "--openvino-device",
        default=OPENVINO_DEVICE,
        help="OpenVINO device, such as CPU, GPU, AUTO, or NPU.",
    )
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Input size for model.")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="NMS IoU threshold.")
    parser.add_argument(
        "--decode",
        choices=("auto", "cpp", "yolov8", "yolov5", "xyxy"),
        default=DECODE_MODE,
        help="Output decode mode.",
    )
    parser.add_argument(
        "--preprocess",
        choices=("resize", "letterbox"),
        default=PREPROCESS_MODE,
        help="Preprocessing mode.",
    )
    parser.add_argument(
        "--classes",
        default=",".join(CLASS_NAMES),
        help="Comma-separated class names.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=SHOW_RESULT,
        help="Show each result in an OpenCV window.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=0,
        help="Delay between images in milliseconds (0 = wait for key press).",
    )
    args = parser.parse_args()
    return args


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            print(f"No images found in: {source}", file=sys.stderr)
        return images
    raise FileNotFoundError(f"Source does not exist: {source}")


def static_dim(value: object, fallback: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def preprocess_resize(image: np.ndarray, input_w: int, input_h: int) -> tuple[np.ndarray, PreprocessMeta]:
    image_h, image_w = image.shape[:2]
    resized = cv2.resize(image, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    meta = PreprocessMeta(
        image_w=image_w,
        image_h=image_h,
        input_w=input_w,
        input_h=input_h,
        mode="resize",
        scale_x=image_w / float(input_w),
        scale_y=image_h / float(input_h),
    )
    return tensor, meta


def preprocess_letterbox(image: np.ndarray, input_w: int, input_h: int) -> tuple[np.ndarray, PreprocessMeta]:
    image_h, image_w = image.shape[:2]
    ratio = min(input_w / image_w, input_h / image_h)
    new_w = int(round(image_w * ratio))
    new_h = int(round(image_h * ratio))
    pad_x = (input_w - new_w) / 2.0
    pad_y = (input_h - new_h) / 2.0

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]
    meta = PreprocessMeta(
        image_w=image_w,
        image_h=image_h,
        input_w=input_w,
        input_h=input_h,
        mode="letterbox",
        scale_x=1.0,
        scale_y=1.0,
        ratio=ratio,
        pad_x=left,
        pad_y=top,
    )
    return tensor, meta


def prepare_output(output: np.ndarray) -> np.ndarray:
    arr = np.asarray(output)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)

    if arr.shape[0] < arr.shape[1] and arr.shape[0] >= 5:
        arr = arr.T
    return arr.astype(np.float32, copy=False)


def cxcywh_to_xyxy(values: np.ndarray) -> np.ndarray:
    boxes = np.empty_like(values[:, :4])
    boxes[:, 0] = values[:, 0] - values[:, 2] * 0.5
    boxes[:, 1] = values[:, 1] - values[:, 3] * 0.5
    boxes[:, 2] = values[:, 0] + values[:, 2] * 0.5
    boxes[:, 3] = values[:, 1] + values[:, 3] * 0.5
    return boxes


def maybe_denormalize_boxes(boxes: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    if boxes.size and float(np.nanmax(np.abs(boxes))) <= 2.0:
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= meta.input_w
        boxes[:, [1, 3]] *= meta.input_h
    return boxes


def map_boxes_to_original(boxes: np.ndarray, meta: PreprocessMeta) -> np.ndarray:
    boxes = maybe_denormalize_boxes(boxes, meta).copy()
    if meta.mode == "letterbox":
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - meta.pad_x) / meta.ratio
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - meta.pad_y) / meta.ratio
    else:
        boxes[:, [0, 2]] *= meta.scale_x
        boxes[:, [1, 3]] *= meta.scale_y

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, meta.image_w - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, meta.image_h - 1)
    return boxes


def choose_auto_decode(stride: int, class_count: int) -> str:
    if stride == 6:
        return "xyxy"
    if class_count > 0 and stride == 4 + class_count:
        return "yolov8"
    if class_count > 0 and stride == 5 + class_count:
        return "yolov5"
    return "cpp"


def decode_predictions(
    raw: np.ndarray,
    meta: PreprocessMeta,
    conf_thres: float,
    decode_mode: str,
    class_count: int,
) -> list[Detection]:
    preds = prepare_output(raw)
    if preds.size == 0 or preds.shape[1] < 6:
        return []

    stride = preds.shape[1]
    mode = choose_auto_decode(stride, class_count) if decode_mode == "auto" else decode_mode

    if mode == "xyxy":
        boxes = preds[:, :4]
        scores = preds[:, 4]
        class_ids = np.rint(preds[:, 5]).astype(np.int32)
    elif mode == "yolov8":
        boxes = cxcywh_to_xyxy(preds)
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
    elif mode == "yolov5":
        boxes = cxcywh_to_xyxy(preds)
        objectness = preds[:, 4]
        class_scores = preds[:, 5:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
        scores = objectness * class_scores[np.arange(class_scores.shape[0]), class_ids]
    else:
        if stride == 6:
            boxes = preds[:, :4]
            scores = preds[:, 4]
            class_ids = np.rint(preds[:, 5]).astype(np.int32)
        else:
            boxes = cxcywh_to_xyxy(preds)
            scores_and_classes = preds[:, 4:]
            best = np.argmax(scores_and_classes, axis=1).astype(np.int32)
            scores = scores_and_classes[np.arange(scores_and_classes.shape[0]), best]
            class_ids = best

    keep = scores >= conf_thres
    if not np.any(keep):
        return []

    boxes = map_boxes_to_original(boxes[keep], meta)
    scores = scores[keep]
    class_ids = class_ids[keep]

    detections: list[Detection] = []
    for box, score, class_id in zip(boxes, scores, class_ids):
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        detections.append(Detection(int(class_id), float(score), box.astype(np.float32)))
    return detections


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return inter / np.maximum(area_a + area_b - inter, 1e-6)


def nms(detections: list[Detection], iou_thres: float) -> list[Detection]:
    if not detections:
        return []
    result: list[Detection] = []
    by_class: dict[int, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    for _, items in by_class.items():
        items = sorted(items, key=lambda d: d.score, reverse=True)
        while items:
            current = items.pop(0)
            result.append(current)
            if not items:
                break
            boxes = np.stack([d.box for d in items], axis=0)
            keep_mask = iou(current.box, boxes) <= iou_thres
            items = [d for d, keep in zip(items, keep_mask) if keep]
    return sorted(result, key=lambda d: d.score, reverse=True)


def bbox_corners(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box.tolist()
    return np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )


def reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    projected = projected.reshape(-1, 2)
    return float(np.mean(np.linalg.norm(projected - image_points, axis=1)))


def solve_xyz(det: Detection) -> tuple[np.ndarray | None, float | None]:
    size = CLASS_SIZES.get(det.class_id)
    if size is None:
        return None, None

    width, height = size
    if width <= 0.0 or height <= 0.0:
        return None, None

    object_points = np.array(
        [
            [-width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, -height / 2.0, 0.0],
            [width / 2.0, height / 2.0, 0.0],
            [-width / 2.0, height / 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    image_points = bbox_corners(det.box)

    candidates: list[tuple[float, np.ndarray]] = []
    flags = [cv2.SOLVEPNP_ITERATIVE]
    sqpnp = getattr(cv2, "SOLVEPNP_SQPNP", None)
    if sqpnp is not None:
        flags.insert(0, sqpnp)
    if abs(width - height) < 1e-6:
        flags.append(cv2.SOLVEPNP_IPPE_SQUARE)

    for flag in flags:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            CAMERA_MATRIX,
            DIST_COEFFS,
            flags=flag,
        )
        if not ok:
            continue
        tvec = tvec.reshape(3)
        if not np.all(np.isfinite(tvec)) or tvec[2] <= 0.0:
            continue
        err = reprojection_error(object_points, image_points, rvec, tvec.reshape(3, 1))
        candidates.append((err, tvec.astype(np.float64)))

    if not candidates:
        return None, None

    best_err, best_tvec = min(candidates, key=lambda item: item[0])
    if best_err > 8.0:
        return None, best_err
    return best_tvec, best_err


def attach_xyz(detections: Sequence[Detection]) -> None:
    for det in detections:
        det.xyz, det.reproj_error = solve_xyz(det)


def color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = (
        (56, 168, 0),
        (0, 165, 255),
        (255, 80, 80),
        (220, 80, 220),
        (60, 180, 230),
    )
    return palette[class_id % len(palette)]


def draw_text_block(
    image: np.ndarray,
    origin: tuple[int, int],
    lines: Sequence[str],
    color: tuple[int, int, int],
    scale: float = 0.55,
) -> None:
    if not lines:
        return
    thickness = 1
    line_height = int(22 * scale / 0.55)
    widths = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] for line in lines]
    box_w = max(widths) + 8
    box_h = line_height * len(lines) + 8
    x, y = origin
    y = max(0, y)
    cv2.rectangle(image, (x, y), (x + box_w, y + box_h), color, -1)
    for idx, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (x + 4, y + 17 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    class_names: Sequence[str],
) -> np.ndarray:
    canvas = image.copy()
    summary_lines = [
        f"detections: {len(detections)}",
    ]
    draw_text_block(canvas, (8, 8), summary_lines, (40, 40, 40), scale=0.58)

    for det in detections:
        x1, y1, x2, y2 = det.box.astype(int).tolist()
        color = color_for_class(det.class_id)
        name = class_names[det.class_id] if 0 <= det.class_id < len(class_names) else str(det.class_id)
        label_lines = [f"{name} {det.score:.2f}"]
        if det.xyz is not None:
            label_lines.extend(
                [
                    f"x={det.xyz[0]:.3f}m",
                    f"y={det.xyz[1]:.3f}m",
                    f"z={det.xyz[2]:.3f}m",
                ]
            )
        elif det.reproj_error is not None:
            label_lines.append(f"PnP err={det.reproj_error:.1f}")
        else:
            label_lines.append("xyz=N/A")

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_y = y1 - 96 if y1 > 104 else y1 + 4
        draw_text_block(canvas, (x1, label_y), label_lines, color, scale=0.52)
    return canvas


def detect_image(runner: Runner, image_path: Path, args: argparse.Namespace, class_names: Sequence[str]) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    # 预处理
    if args.preprocess == "letterbox":
        tensor, meta = preprocess_letterbox(image, runner.input_w, runner.input_h)
    else:
        tensor, meta = preprocess_resize(image, runner.input_w, runner.input_h)

    # 推理
    raw_output = runner.infer(tensor)

    # 后处理
    detections = decode_predictions(
        raw_output,
        meta=meta,
        conf_thres=args.conf,
        decode_mode=args.decode,
        class_count=len(class_names),
    )
    detections = nms(detections, args.iou)
    attach_xyz(detections)

    # 绘制结果
    result = draw_detections(image, detections, class_names)
    
    # 打印检测信息
    print(f"\n{image_path.name}: Found {len(detections)} objects")
    for det in detections:
        x1, y1, x2, y2 = det.box.tolist()
        label = class_names[det.class_id] if 0 <= det.class_id < len(class_names) else str(det.class_id)
        if det.xyz is None:
            xyz = "xyz=N/A"
        else:
            xyz = f"xyz=({det.xyz[0]:.3f},{det.xyz[1]:.3f},{det.xyz[2]:.3f})m"
        print(f"  {label}: {det.score:.3f}, box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), {xyz}")
    
    return result


def main() -> int:
    args = parse_args()
    
    if not args.source.exists():
        print(f"Source does not exist: {args.source}", file=sys.stderr)
        return 1

    class_names = [name.strip() for name in args.classes.split(",") if name.strip()]
    images = collect_images(args.source.resolve())
    if not images:
        print(f"No images found in: {args.source}", file=sys.stderr)
        return 1

    # 初始化推理后端
    runners: list[Runner] = []
    if args.backend in ("onnx", "both"):
        print("Loading ONNX Runtime backend...")
        runners.append(ONNXRunner(args.model.resolve(), args.imgsz))
    if args.backend in ("openvino", "both"):
        print("Loading OpenVINO backend...")
        runners.append(
            OpenVINORunner(
                args.openvino_model.resolve(),
                args.openvino_device,
                args.imgsz,
            )
        )
    
    if not runners:
        print("No backend selected.", file=sys.stderr)
        return 1

    print(f"\nProcessing {len(images)} images...")
    print(f"Backend: {args.backend}, decode: {args.decode}, preprocess: {args.preprocess}")
    print(f"Confidence threshold: {args.conf}, IoU threshold: {args.iou}")
    print("-" * 60)

    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Processing: {image_path.name}")
        
        for runner in runners:
            start_time = time.perf_counter()
            result_image = detect_image(runner, image_path, args, class_names)
            inference_time = (time.perf_counter() - start_time) * 1000
            
            print(f"  [{runner.name}] Inference time: {inference_time:.2f} ms")
            
            if args.show:
                # 在窗口标题中显示图像名称和后端信息
                window_title = f"{image_path.name} - {runner.name}"
                cv2.imshow(window_title, result_image)
                
                # 等待按键，如果设置了delay则自动切换
                key = cv2.waitKey(args.delay)
                if key in (27, ord('q')):  # ESC 或 q 退出
                    cv2.destroyAllWindows()
                    return 0
                elif key == ord('s'):  # s 保存当前图像
                    save_path = image_path.parent / f"{image_path.stem}_detected{image_path.suffix}"
                    cv2.imwrite(str(save_path), result_image)
                    print(f"    Saved to: {save_path}")
                
                cv2.destroyWindow(window_title)

    if args.show:
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())