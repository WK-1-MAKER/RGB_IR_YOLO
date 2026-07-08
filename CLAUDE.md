# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- **不做冗余开发**: 新增功能前先检查本项目已有模块是否已提供相同或相近能力，优先复用而非重复实现
- **开发过程中禁止自动提交代码**:修改代码过程中禁止通过git提交代码，禁止git add，git commit, git push等操作，代码的git管理统一由人工完成。
- **未经允许不可修改代码**:每次修改代码前都要询问一次，未经允许不可直接修改代码
- **可以使用虚拟conda环境**:本项目的conda环境是/home/SENSETIME/wenkai/envs/yolov8

## Python Framework

### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
python train.py --data data.yaml --cfg models/yolov8n-two_stream.yaml --epochs 100 --batch-size 32
```

### Test / Evaluate
```bash
python test.py --weights weights/best.pt --data data.yaml
```

### Inference
```bash
python detect.py --source1 path/to/rgb/images --source2 path/to/ir/images --weights weights/best.pt
```

### Export to ONNX
```bash
python export_two_stream.py --weights weights/best.pt --img 640 640 --opset 12
```

## ROS C++ Package (`ROS/src/two_stream/`)

### Build
```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make -DCMAKE_BUILD_TYPE=Release
```

### Run
```bash
source devel/setup.bash
roslaunch two_stream yolo_topic.launch
```

### Tests
```bash
cd /home/SENSETIME/wenkai/YOLOV8_IR_RGB/ROS
catkin_make run_tests

# Individual binaries after build
./devel/lib/two_stream/test_yolov8_decode
./devel/lib/two_stream/test_detector_postprocess
./devel/lib/two_stream/test_read_config
./devel/lib/two_stream/test_camera_geometry
```

## Architecture

This project implements Cross-Modality Fusion Transformer (CFT) for multispectral object detection by combining RGB and thermal (IR) image pairs.

### Python Training Pipeline

- **Entry points**: `train.py` (training), `test.py` (evaluation), `detect.py` (inference)
- **Model definition**: `models/yolo.py` parses YAML configs and builds layers; transformer fusion blocks live in `ultralytics/nn/modules/transformer.py`
- **Dual-input data loading**: `utils/datasets.py` reads synchronized RGB+IR pairs for LLVIP, FLIR, and VEDAI datasets
- **Model configs**: `models/` and `ultralytics/cfg/models/v8/` contain YAML architecture files. The paper uses `*_transformerx3_dataset.yaml` variants.
- **Dataset config**: `data.yaml` and `data/multispectral/` define RGB+IR train/val paths and class names (1 class: person for LLVIP)

### ROS C++ Inference Pipeline

The ROS1 catkin C++14 package (`ROS/src/two_stream/`) runs real-time detection:

1. **`yolo_node.cpp`** — subscribes to synchronized RGB/IR topics via `message_filters` ApproximateTime; orchestrates the full pipeline; publishes annotated output
2. **`detector.cpp` + `yolov8_decode.cpp`** — runs the dual-input ONNX model via ONNX Runtime GPU; decodes rank-3 output and applies NMS
3. **`super_point.cpp`** — TensorRT SuperPoint keypoint detector + 258-dim descriptors; builds `.engine` from `.onnx` on first run
4. **`light_glue.cpp`** — TensorRT LightGlue matcher; cross-modal RGB↔IR feature matching
5. **Stereo geometry** (in `yolo_node.cpp`) — loads calibration from `config/config.yaml`; computes homography (H) and fundamental matrix (F); triangulates 3D positions

**External dependency paths (hardcoded in `CMakeLists.txt`):**
- ONNX Runtime 1.18.0: `/home/SENSETIME/wenkai/3rd/onnxruntime-linux-x64-gpu-1.18.0`
- TensorRT 8.6.1.6: `/home/SENSETIME/wenkai/3rd/TensorRT-8.6.1.6`
- CUDA 11.8

**Model files (git-ignored, must be placed manually):**
- `ROS/src/two_stream/models/best.onnx` — dual-input YOLOv8 (189 MB)
- `ROS/src/two_stream/weights/superpoint_v1.{onnx,engine}`
- `ROS/src/two_stream/weights/superpoint_lightglue.{onnx,engine}`

TensorRT `.engine` files are cached; delete them to force rebuild if hardware changes.

**Key data structures:**
- `Detection` (`yolo_utils.h`): `cv::Rect box`, `float conf`, `int classId`
- `Configs` / `CameraCalibrationConfig` (`read_config.h`): parsed from `config/config.yaml`

**Configurable launch params** (`launch/yolo_topic.launch`): `rgb_topic`, `ir_topic`, `conf_threshold` (0.30), `iou_threshold` (0.40), `model_path`, `config_path`, `model_dir`, `input_width/height` (640×640), `use_gpu`, `show_window`


