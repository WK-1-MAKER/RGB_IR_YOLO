# 训练
python train.py 

# 导出onnx
python export_two_stream.py \
    --weights uns/train/transformer-aifi-s/weights/best.pt \
    --simplify \
    --dtype auto

# 导出trt
TRT_ROOT=/home/SENSETIME/wenkai/3rd/TensorRT-8.6.1.6
export LD_LIBRARY_PATH=$TRT_ROOT/targets/x86_64-linux-gnu/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

$TRT_ROOT/targets/x86_64-linux-gnu/bin/trtexec \
  --onnx=runs/train/transformer-aifi-s/weights/aifi-best.onnx \
  --saveEngine=runs/train/transformer-aifi-s/weights/aifi-best-fp16.engine \
  --fp16 \
  --workspace=4096 \

# 测试trt
$TRT_ROOT/targets/x86_64-linux-gnu/bin/trtexec \
  --loadEngine=runs/train/transformer-aifi-s/weights/aifi-best-fp16.engine \
  --shapes=images_rgb:1x3x640x640,images_ir:1x3x640x640 \
  --warmUp=1000 \
  --duration=30 \
  --iterations=1000 \
  --useSpinWait