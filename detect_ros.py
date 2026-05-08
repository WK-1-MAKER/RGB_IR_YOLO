# usr/bin/env python3
import argparse
import time
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

# ROS imports
import rospy
import message_filters
from sensor_msgs.msg import Image
# 移除 cv_bridge 导入，避免库冲突
# from cv_bridge import CvBridge, CvBridgeError 

# YOLO imports
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized

class DualStreamDetector:
    def __init__(self, opt):
        self.opt = opt
        # self.bridge = CvBridge() # 移除
        
        # Initialize ROS node
        rospy.init_node('dual_stream_detector', anonymous=True)
        
        # Initialize Model
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        
        print(f"Loading model from {opt.weights}...")
        self.model = attempt_load(opt.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(opt.img_size, s=self.stride)  # check img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        
        if self.half:
            self.model.half()  # to FP16
        
        cudnn.benchmark = True  # set True to speed up constant image size inference

        self.count = 0
            
        # Warmup
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())), 
                       torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))

        # Publishers
        self.pub_rgb = rospy.Publisher('/rgb_result', Image, queue_size=1)
        self.pub_ir = rospy.Publisher('/ir_result', Image, queue_size=1)
        
        # Subscribers with Synchronization
        # 注意：请确保这里的话题名称与你实际发送的一致
        rgb_sub = message_filters.Subscriber('/pub_rgb', Image)
        ir_sub = message_filters.Subscriber('/pub_t', Image)
        
        # slop 参数决定了允许的时间误差（秒）
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, ir_sub], queue_size=10, slop=0.1)
        ts.registerCallback(self.callback)

        print("ROS Node initialized. Waiting for images...")

    def imgmsg_to_cv2(self, img_msg):
        """
        手动将 ROS Image 消息转换为 OpenCV 图像 (numpy array)，替代 cv_bridge
        """
        dtype = np.uint8
        n_channels = 3
        
        # 简单处理 bgr8 和 rgb8
        if img_msg.encoding == 'bgr8' or img_msg.encoding == 'rgb8':
            img_buf = np.frombuffer(img_msg.data, dtype=dtype)
            img = np.reshape(img_buf, (img_msg.height, img_msg.width, n_channels))
            
            # 如果是 rgb8，转为 bgr8 供 opencv 使用
            if img_msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img.copy()
        elif img_msg.encoding == 'mono8':
            img_buf = np.frombuffer(img_msg.data, dtype=dtype)
            img = np.reshape(img_buf, (img_msg.height, img_msg.width))
            # 转为 3 通道 BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img.copy()
        else:
            rospy.logwarn(f"Unsupported encoding: {img_msg.encoding}. Assuming bgr8.")
            img_buf = np.frombuffer(img_msg.data, dtype=dtype)
            try:
                img = np.reshape(img_buf, (img_msg.height, img_msg.width, n_channels))
                return img.copy()
            except Exception as e:
                rospy.logerr(f"Failed to reshape image: {e}")
                return None

    def cv2_to_imgmsg(self, cv_image):
        """
        手动将 OpenCV 图像转换为 ROS Image 消息，替代 cv_bridge
        """
        img_msg = Image()
        img_msg.header.stamp = rospy.Time.now()
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = "bgr8"
        img_msg.is_bigendian = 0
        img_msg.step = cv_image.shape[1] * 3
        img_msg.data = cv_image.tobytes()
        return img_msg

    def preprocess_img(self, img0):
        # Padded resize
        img = cv2.resize(img0, (self.imgsz, self.imgsz))
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def callback(self, rgb_msg, ir_msg):
        t1 = time_synchronized()
        try:
            # 1. Convert ROS messages to OpenCV images (Manual)
            im0_rgb = self.imgmsg_to_cv2(rgb_msg)
            im0_ir = self.imgmsg_to_cv2(ir_msg)
            
            if im0_rgb is None or im0_ir is None:
                return

            # 2. Preprocess
            img_rgb = self.preprocess_img(im0_rgb)
            img_ir = self.preprocess_img(im0_ir)
            
            # 3. Inference
            with torch.no_grad():
                pred = self.model(img_rgb, img_ir, augment=self.opt.augment)[0]
                
            # 4. NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            
            # 5. Process detections
            det = pred[0] # batch size is 1
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img_rgb.shape[2:], det[:, :4], im0_rgb.shape).round()

                # Draw boxes
                for *xyxy, conf, cls in reversed(det):
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0_rgb, label=label, color=colors(int(cls), True), line_thickness=3)
                    plot_one_box(xyxy, im0_ir, label=label, color=colors(int(cls), True), line_thickness=3)
            
            # 6. Publish results (Manual)
            self.pub_rgb.publish(self.cv2_to_imgmsg(im0_rgb))
            self.pub_ir.publish(self.cv2_to_imgmsg(im0_ir))

        except Exception as e:
            print(f"Error in callback: {e}")
            import traceback
            traceback.print_exc()
        t2 = time_synchronized()
        print(f'Inference and publishing done. ({t2 - t1:.6f}s, {1/(t2 - t1):.6f}Hz)')
        self.count += 1
        print(f'Processed frame count: {self.count}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/train_over1/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    opt = parser.parse_args()
    
    try:
        detector = DualStreamDetector(opt)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass