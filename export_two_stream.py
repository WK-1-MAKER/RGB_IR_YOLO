import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(Path(__file__).parent.absolute().__str__())

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import check_img_size, check_requirements, file_size, set_logging
from utils.torch_utils import select_device


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
    model = attempt_load(opt.weights, map_location=device)
    model.eval()

    gs = int(max(model.stride))
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]
    img_rgb = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)
    img_ir = torch.zeros(opt.batch_size, 3, *opt.img_size).to(device)

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
