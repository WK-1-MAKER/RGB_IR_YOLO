import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(Path(__file__).parent.absolute().__str__())

import models
from utils.activations import Hardswish, SiLU
from utils.general import check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device


SUPPORTED_EXPORT_DTYPES = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def floating_state_dtypes(model):
    """Return the unique floating-point dtypes used by model parameters/buffers."""
    tensors = list(model.parameters()) + list(model.buffers())
    return {tensor.dtype for tensor in tensors if tensor.is_floating_point()}


def resolve_export_dtype(model, requested):
    """Resolve an explicit or checkpoint-derived ONNX export dtype."""
    if requested in SUPPORTED_EXPORT_DTYPES:
        return SUPPORTED_EXPORT_DTYPES[requested]
    if requested != "auto":
        raise ValueError("unsupported export dtype: %s" % requested)

    dtypes = floating_state_dtypes(model)
    if not dtypes:
        raise ValueError("checkpoint has no floating-point parameters or buffers")
    if len(dtypes) != 1:
        names = ", ".join(sorted(str(dtype) for dtype in dtypes))
        raise ValueError(
            "mixed floating-point dtypes in checkpoint: %s; "
            "use --dtype fp16 or --dtype fp32" % names
        )

    dtype = next(iter(dtypes))
    if dtype not in SUPPORTED_EXPORT_DTYPES.values():
        raise ValueError(
            "unsupported checkpoint floating dtype: %s; "
            "use --dtype fp16 or --dtype fp32" % dtype
        )
    return dtype


def load_model_for_export(weights, map_location):
    """Load a checkpoint model without an implicit FP32 cast."""
    checkpoint = torch.load(weights, map_location=map_location)
    model = checkpoint["ema"] if checkpoint.get("ema") is not None else checkpoint["model"]
    model = model.to(map_location)
    return model.eval()


def fuse_model_for_export(model, target_dtype):
    """Fuse layers in FP32, then restore the requested export dtype."""
    model = model.float()
    if hasattr(model, "fuse"):
        model = model.fuse()
    return model.to(dtype=target_dtype).eval()


def make_export_inputs(batch_size, img_size, device, dtype):
    """Create dtype-aligned RGB and IR tracing inputs."""
    img_rgb = torch.zeros(batch_size, 3, *img_size, device=device, dtype=dtype)
    return img_rgb, torch.zeros_like(img_rgb)


class ExportInterpolate(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)


def parse_opt(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train/exp7/weights/best.pt")
    parser.add_argument("--img-size", nargs="+", type=int, default=[640, 640])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action="store_true")
    parser.add_argument("--dtype", choices=["auto", "fp16", "fp32"], default="auto")
    opt = parser.parse_args(args=args)
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1
    return opt


def make_onnx_path(weights):
    return str(Path(weights).with_suffix(".onnx"))


def build_dynamic_axes(dynamic):
    if not dynamic:
        return None
    return {
        "images_rgb": {0: "batch", 2: "height", 3: "width"},
        "images_ir": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch"},
    }


def prepare_model_for_export(model):
    for _, module in model.named_modules():
        if hasattr(module, "_non_persistent_buffers_set"):
            module._non_persistent_buffers_set = set()
        if isinstance(module, models.common.Conv):
            if isinstance(module.act, nn.Hardswish):
                module.act = Hardswish()
            elif isinstance(module.act, nn.SiLU):
                module.act = SiLU()
        elif isinstance(module, models.common.GPT):
            module.avgpool = ExportInterpolate((module.vert_anchors, module.horz_anchors))
    model.model[-1].export = True
    model.model[-1].format = "onnx"


def export_onnx(opt):
    set_logging()
    t = time.time()
    device = select_device(opt.device)
    model = load_model_for_export(opt.weights, map_location=device)
    target_dtype = resolve_export_dtype(model, opt.dtype)
    model = fuse_model_for_export(model, target_dtype)
    print(f"Export dtype: {target_dtype}")

    gs = int(max(model.stride))
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]
    img_rgb, img_ir = make_export_inputs(opt.batch_size, opt.img_size, device, target_dtype)

    prepare_model_for_export(model)

    for _ in range(2):
        model(img_rgb, img_ir)

    import onnx

    out_path = make_onnx_path(opt.weights)
    torch.onnx.export(
        model,
        (img_rgb, img_ir),
        out_path,
        verbose=False,
        opset_version=13,
        input_names=["images_rgb", "images_ir"],
        output_names=["output"],
        dynamic_axes=build_dynamic_axes(opt.dynamic),
    )

    model_onnx = onnx.load(out_path)
    onnx.checker.check_model(model_onnx)

    if opt.simplify:
        check_requirements(["onnx-simplifier"])
        import onnxsim

        model_onnx, check = onnxsim.simplify(
            model_onnx,
            dynamic_input_shape=opt.dynamic,
            input_shapes={
                "images_rgb": list(img_rgb.shape),
                "images_ir": list(img_ir.shape),
            } if opt.dynamic else None,
        )
        assert check, "assert check failed"
        onnx.save(model_onnx, out_path)

    print(f"ONNX export success: {out_path} ({file_size(out_path):.1f} MB) in {time.time() - t:.2f}s")


def main(args=None):
    opt = parse_opt(args=args)
    export_onnx(opt)


if __name__ == "__main__":
    main()
