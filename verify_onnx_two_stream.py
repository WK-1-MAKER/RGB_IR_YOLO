import argparse
from pathlib import Path

import numpy as np


def parse_opt(args=None):
    parser = argparse.ArgumentParser(description="Verify dual-stream ONNX deployability with ONNX Runtime")
    parser.add_argument("--onnx", required=True, help="path to ONNX model")
    parser.add_argument("--image-rgb", required=True, help="path to RGB image")
    parser.add_argument("--image-ir", required=True, help="path to IR image")
    parser.add_argument("--img-size", type=int, default=640, help="inference image size")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["CPUExecutionProvider"],
        help="onnxruntime execution providers",
    )
    return parser.parse_args(args=args)


def validate_input_names(input_names):
    required = ("images_rgb", "images_ir")
    missing = [name for name in required if name not in input_names]
    if missing:
        raise ValueError(f"missing expected ONNX inputs: {', '.join(missing)}")
    return required


def read_image(path):
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required to read verification images") from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    return image


def preprocess_image(image, img_size, letterbox_fn=None):
    if image is None:
        raise ValueError("image is None")

    if letterbox_fn is None:
        from utils.datasets import letterbox as letterbox_fn

    image = letterbox_fn(image, img_size, stride=32, auto=False)[0]
    image = image[:, :, ::-1].transpose(2, 0, 1)
    image = np.ascontiguousarray(image, dtype=np.float32)
    image /= 255.0
    return np.expand_dims(image, axis=0)


def create_session(onnx_path, providers):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError("onnxruntime is required for ONNX deployability verification") from exc

    return ort.InferenceSession(str(onnx_path), providers=providers)


def describe_value(value):
    if isinstance(value, np.ndarray):
        return f"shape={tuple(value.shape)}, dtype={value.dtype}"
    return str(value)


def print_session_metadata(session, outputs):
    print(f"Providers: {session.get_providers()}")
    print("Inputs:")
    for node in session.get_inputs():
        print(f"  - {node.name}: shape={node.shape}, type={node.type}")

    print("Outputs:")
    for node in session.get_outputs():
        print(f"  - {node.name}: shape={node.shape}, type={node.type}")

    print("Runtime Outputs:")
    for idx, output in enumerate(outputs):
        print(f"  - output[{idx}]: {describe_value(output)}")

    first = outputs[0]
    if isinstance(first, np.ndarray) and first.size > 0:
        print(f"First Output Stats: min={float(first.min()):.6f}, max={float(first.max()):.6f}")


def verify_onnx(opt):
    session = create_session(opt.onnx, opt.providers)

    input_names = [node.name for node in session.get_inputs()]
    rgb_name, ir_name = validate_input_names(input_names)

    image_rgb = read_image(opt.image_rgb)
    image_ir = read_image(opt.image_ir)
    tensor_rgb = preprocess_image(image_rgb, opt.img_size)
    tensor_ir = preprocess_image(image_ir, opt.img_size)

    outputs = session.run(None, {rgb_name: tensor_rgb, ir_name: tensor_ir})
    if not outputs:
        raise RuntimeError("onnxruntime returned no outputs")

    return session, outputs


def main(args=None):
    opt = parse_opt(args=args)
    print(f"Model: {opt.onnx}")

    try:
        session, outputs = verify_onnx(opt)
    except Exception as exc:
        print(f"[FAIL] ONNX deployability check failed: {exc}")
        raise SystemExit(1) from exc

    print_session_metadata(session, outputs)
    print(f"[OK] ONNX deployability check passed: {opt.onnx}")


if __name__ == "__main__":
    main()
