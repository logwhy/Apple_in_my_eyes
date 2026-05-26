#!/usr/bin/env python3
"""Run last.onnx on images for quick Windows-side validation.

Dependencies:
  pip install onnxruntime opencv-python numpy

Examples:
  1. Edit SOURCE_PATH below, then run:
     python smarthome_vision_ros2/test_last_onnx.py

  2. Or pass a path from terminal:
  python smarthome_vision_ros2/test_last_onnx.py path\\to\\image.jpg
  python smarthome_vision_ros2/test_last_onnx.py path\\to\\images --show
  python smarthome_vision_ros2/test_last_onnx.py img.jpg --conf 0.35 --decode yolov8
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    import cv2
    import numpy as np
    import onnxruntime as ort
except ImportError as exc:
    raise SystemExit(
        f"Missing dependency: {exc.name}\n"
        "Install it with: pip install onnxruntime opencv-python numpy"
    ) from exc


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SCRIPT_DIR = Path(__file__).resolve().parent

# Change this path when testing on Windows. It can be one image or a directory.
SOURCE_PATH = r"C:\Users\38480\Desktop\lf_sentry-main\smarthome_vision_ros2\assets\2.png"

MODEL_PATH = SCRIPT_DIR / "assets" / "last.onnx"
OUTPUT_DIR = SCRIPT_DIR / "result" / "last_onnx_test"

IMG_SIZE = 640
CONF_THRESHOLD = 0.75
IOU_THRESHOLD = 0.45
DECODE_MODE = "auto"  # auto, cpp, yolov8, yolov5, xyxy
PREPROCESS_MODE = "resize"  # resize matches the current C++ detector; letterbox often matches YOLO.
CLASS_NAMES = ["0", "1", "2"]
SHOW_RESULT = False
SAVE_TXT = False

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

# Width/height in meters, copied from config/vision.yaml.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test smarthome_vision_ros2/assets/last.onnx on images."
    )
    parser.add_argument(
        "source",
        type=Path,
        nargs="?",
        default=None,
        help="Image file or image directory. If omitted, SOURCE_PATH at the top of this file is used.",
    )
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to ONNX model.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--imgsz", type=int, default=IMG_SIZE, help="Fallback input size for dynamic ONNX.")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="NMS IoU threshold.")
    parser.add_argument(
        "--decode",
        choices=("auto", "cpp", "yolov8", "yolov5", "xyxy"),
        default=DECODE_MODE,
        help=(
            "Output decode mode. cpp mirrors the ROS2 C++ detector convention; "
            "yolov8 expects [cx,cy,w,h,cls...]; yolov5 expects [cx,cy,w,h,obj,cls...]; "
            "xyxy expects [x1,y1,x2,y2,score,class]."
        ),
    )
    parser.add_argument(
        "--preprocess",
        choices=("resize", "letterbox"),
        default=PREPROCESS_MODE,
        help="resize matches the current C++ detector; letterbox often matches YOLO training.",
    )
    parser.add_argument(
        "--classes",
        default=",".join(CLASS_NAMES),
        help="Comma-separated class names. Default matches config/vision.yaml class ids.",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        default=SAVE_TXT,
        help="Save detections as txt files.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=SHOW_RESULT,
        help="Show each result in an OpenCV window.",
    )
    args = parser.parse_args()
    if args.source is None and SOURCE_PATH.strip():
        args.source = Path(SOURCE_PATH)
    return args


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    raise FileNotFoundError(f"Source does not exist: {source}")


def static_dim(value: object, fallback: int) -> int:
    if isinstance(value, int) and value > 0:
        return value
    return fallback


def make_session(model_path: Path) -> ort.InferenceSession:
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {model_path}")
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def resolve_input_size(session: ort.InferenceSession, fallback: int) -> tuple[str, int, int]:
    input_meta = session.get_inputs()[0]
    shape = input_meta.shape
    if len(shape) != 4:
        raise ValueError(f"Expected NCHW 4D input, got shape: {shape}")
    input_h = static_dim(shape[2], fallback)
    input_w = static_dim(shape[3], fallback)
    return input_meta.name, input_w, input_h


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
    elapsed_ms: float,
) -> np.ndarray:
    canvas = image.copy()
    summary_lines = [
        f"time: {elapsed_ms:.2f} ms",
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


def save_txt(path: Path, detections: Sequence[Detection], image_w: int, image_h: int) -> None:
    lines = []
    for det in detections:
        x1, y1, x2, y2 = det.box.tolist()
        cx = ((x1 + x2) * 0.5) / image_w
        cy = ((y1 + y2) * 0.5) / image_h
        w = (x2 - x1) / image_w
        h = (y2 - y1) / image_h
        if det.xyz is None:
            xyz = "nan nan nan"
        else:
            xyz = f"{det.xyz[0]:.6f} {det.xyz[1]:.6f} {det.xyz[2]:.6f}"
        lines.append(f"{det.class_id} {det.score:.6f} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {xyz}")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_image(
    session: ort.InferenceSession,
    input_name: str,
    image_path: Path,
    input_w: int,
    input_h: int,
    args: argparse.Namespace,
    class_names: Sequence[str],
) -> tuple[np.ndarray, list[Detection], float]:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    start = time.perf_counter()
    if args.preprocess == "letterbox":
        tensor, meta = preprocess_letterbox(image, input_w, input_h)
    else:
        tensor, meta = preprocess_resize(image, input_w, input_h)

    outputs = session.run(None, {input_name: tensor})
    detections = decode_predictions(
        outputs[0],
        meta=meta,
        conf_thres=args.conf,
        decode_mode=args.decode,
        class_count=len(class_names),
    )
    detections = nms(detections, args.iou)
    attach_xyz(detections)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    result = draw_detections(image, detections, class_names, elapsed_ms)
    return result, detections, elapsed_ms


def print_model_info(session: ort.InferenceSession, input_w: int, input_h: int) -> None:
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()
    print(f"Model input : {input_meta.name} {input_meta.shape} -> using {input_w}x{input_h}")
    for item in output_meta:
        print(f"Model output: {item.name} {item.shape}")


def main() -> int:
    args = parse_args()
    if args.source is None:
        print(
            "Please edit SOURCE_PATH at the top of smarthome_vision_ros2/test_last_onnx.py "
            "or pass an image path once from the terminal.",
            file=sys.stderr,
        )
        return 1

    class_names = [name.strip() for name in args.classes.split(",") if name.strip()]
    session = make_session(args.model.resolve())
    input_name, input_w, input_h = resolve_input_size(session, args.imgsz)
    images = collect_images(args.source.resolve())
    if not images:
        print(f"No images found in: {args.source}", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    print_model_info(session, input_w, input_h)
    print(f"Images: {len(images)}")
    print(f"Decode: {args.decode}, preprocess: {args.preprocess}, conf: {args.conf}, iou: {args.iou}")

    for image_path in images:
        result, detections, elapsed_ms = run_image(
            session=session,
            input_name=input_name,
            image_path=image_path,
            input_w=input_w,
            input_h=input_h,
            args=args,
            class_names=class_names,
        )
        out_path = args.output / f"{image_path.stem}_det{image_path.suffix}"
        ok, encoded = cv2.imencode(out_path.suffix, result)
        if not ok:
            raise RuntimeError(f"Failed to encode result image: {out_path}")
        encoded.tofile(str(out_path))

        if args.save_txt:
            save_txt(out_path.with_suffix(".txt"), detections, result.shape[1], result.shape[0])

        print(f"{image_path.name}: {len(detections)} detections, {elapsed_ms:.2f} ms -> {out_path}")
        for det in detections:
            x1, y1, x2, y2 = det.box.tolist()
            label = class_names[det.class_id] if 0 <= det.class_id < len(class_names) else str(det.class_id)
            if det.xyz is None:
                xyz = "xyz=N/A"
            else:
                xyz = f"xyz=({det.xyz[0]:.3f},{det.xyz[1]:.3f},{det.xyz[2]:.3f})m"
            print(f"  {label}: {det.score:.3f}, box=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), {xyz}")

        if args.show:
            cv2.imshow("last.onnx result", result)
            key = cv2.waitKey(0)
            if key in (27, ord("q")):
                break

    if args.show:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
