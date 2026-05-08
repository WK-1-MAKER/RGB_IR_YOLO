#!/usr/bin/env python3
# filepath: /home/wk/multispectral-object-detection-main/detect_ros.py

import argparse
import time
import sys
from pathlib import Path
import threading
import queue

import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

import matplotlib.pyplot as plt

# ROS imports
import rospy
import message_filters
from sensor_msgs.msg import Image

#LightGlue imports
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd, numpy_image_to_torch
from lightglue import viz2d

# YOLO imports
from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, time_synchronized
from some_function import align_images, realign_boxes, compute_fundamental_matrix, imgmsg_to_cv2, cv2_to_imgmsg
from some_function import triangulate_refine_gauss_newton, generate_mask, draw_matches, correct_points

class DualStreamDetector:
    def __init__(self, opt):
        self.opt = opt
        
        # Initialize ROS node
        rospy.init_node('dual_stream_detector', anonymous=True)
        
        # Initialize Model
        self.device = select_device(opt.device)  #选择设备
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA 半精度仅支持CUDA
        
        print(f"Loading model from {opt.weights}...")

        #双模态yolo模型
        self.model = attempt_load(opt.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(opt.img_size, s=self.stride)  # check img_size 检查 img_size
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        
        #superpooint + lightglue模型
        self.extractor = SuperPoint(max_num_keypoints=1024).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint", 
                                 depth_confidence=0.9, 
                                 width_confidence=0.9).eval().to(self.device)
        
        if self.half:
            self.model.half()  # to FP16 半精度转换
        
        # 对于动态输入尺寸（裁减后的图像大小不一），一定要关闭 benchmark
        # 否则每一帧都会重新搜索最优算法，导致极大的延迟
        cudnn.benchmark = False 
        self.count = 0
            
        # Warmup 输入圈黑图片跑一轮，热身
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())), 
                       torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))
        
        # --- 多线程设置 ---
        # maxsize=1 意味着如果推理线程在忙，队列满了，新的帧会被丢弃（保证低延迟）
        # queue.Queue是线程安全的队列，因为它是同一个类里的成员变量，所以多线程可以共享
        self.input_queue = queue.Queue(maxsize=1) 

        self.K_rgb = np.array([[606.97900390625, 0, 320.3143615722656],
                   [0, 607.1318359375, 247.97427368164062],
                   [0, 0, 1]])
        self.K_ir = np.array([
            [593.946114205786, 0, 322.242307522658],  # fx, 0, cx
            [0, 592.545920338569, 264.298408236226], # 0, fy, cy
            [0, 0, 1]])
        self.R = np.array([
                    [1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]])
        self.t = np.array([[-0.036], [0.03], [-0.009]])  #单位：米

        self.H = self.K_ir @ (self.R - self.t @ np.array([[0,0,1]]) / 3.0) @ np.linalg.inv(self.K_rgb)  #单应矩阵 H = K_ir * R * K_rgb_inv
        self.H_inv = np.linalg.inv(self.H)
        self.K_rgb_inv = np.linalg.inv(self.K_rgb)
        self.K_ir_inv = np.linalg.inv(self.K_ir)

        # 计算基础矩阵 F
        # 注意：H = K2 @ (R - t_vec/d) @ K1_inv
        # 而 Homography 对应的物理变换通常是 P2 = R P1 + t_vec * (factor)
        # 既然 H 用的是 -t，说明物理位移对于 F 的定义 (P2 = R P1 + T) 来说应该是 -self.t
        self.F = compute_fundamental_matrix(self.K_rgb_inv, self.K_ir_inv, self.R, self.t)

        # Subscribers
        rgb_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        ir_sub = message_filters.Subscriber('/pub_ir', Image)
        
        ts = message_filters.ApproximateTimeSynchronizer([rgb_sub, ir_sub], queue_size=10, slop=0.1) #时间同步器，队列10，时间差0.1s
        ts.registerCallback(self.callback)

        # Publishers
        self.pub_rgb = rospy.Publisher('/rgb_result', Image, queue_size=1) #队列以1大小发布
        self.pub_ir = rospy.Publisher('/ir_result', Image, queue_size=1) 
        self.pub_features = rospy.Publisher('/matched_features', Image, queue_size=1)   

         # 启动推理线程
        self.worker_thread = threading.Thread(target=self.inference_loop, daemon=True)
        self.worker_thread.start()

        print("ROS Node initialized. Waiting for images...")

    def preprocess_img_cpu(self, img0):
        """
        CPU 端的预处理：Resize 和 Transpose
        返回 numpy array，不涉及 GPU 操作
        """
        # Padded resize
        img = cv2.resize(img0, (self.imgsz, self.imgsz))
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB （H, W, C) to (C, H, W)
        img = np.ascontiguousarray(img)
        return img

    def preprocess_img_gpu(self, img_numpy):
        """
        GPU 端的预处理：转 Tensor，归一化
        """
        img = torch.from_numpy(img_numpy).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    #-----------这两个是lightglue需要的图像预处理函数--------------
    def numpy_image_to_torch(self, image: np.ndarray) -> torch.Tensor:
        """Normalize the image tensor and reorder the dimensions."""
        if image.ndim == 3:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        elif image.ndim == 2:
            image = image[None]  # add channel axis
        else:
            raise ValueError(f"Not an image: {image.shape}")
        image_tensor = torch.from_numpy(image)
        #image_tensor = image_tensor.half() if self.half else image_tensor.float()
        image_tensor = image_tensor.float()
        return image_tensor / 255.0
    
    def preprocess_feature_detect_image(self, cv_image):
        """
        Convert OpenCV image (BGR) to torch tensor (RGB, 1, C, H, W) normalized
        Returns: numpy_image (RGB), torch_tensor
        """
        # Convert BGR to RGB for LightGlue/SuperPoint
        image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image_tensor = numpy_image_to_torch(image_rgb)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        return image_tensor
    #-------------------------------------------------------

    def callback(self, rgb_msg, ir_msg):
        """
        生产者：接收消息，CPU预处理，放入队列
        """
        try:
            # 1. Convert ROS messages to OpenCV images
            im0_rgb = imgmsg_to_cv2(rgb_msg)  #此时还是BGR格式
            im0_ir = imgmsg_to_cv2(ir_msg)            

            if im0_rgb is None or im0_ir is None:
                return
            
            im0_rgb_aligned = align_images(im0_rgb, im0_ir, self.H)
            # 2. CPU Preprocess (Resize, Transpose)
            img_rgb_np = self.preprocess_img_cpu(im0_rgb_aligned)  #裁减到imgsz大小,BGRtoRGB
            img_ir_np = self.preprocess_img_cpu(im0_ir)
            
            # 3. Put into Queue
            # 如果队列满了，说明推理线程处理不过来，直接丢弃当前帧 (put_nowait 会抛出 Full 异常)
            try:
                self.input_queue.put_nowait({
                    'img_rgb_np': img_rgb_np,  #im0是原图，img是预处理后的图
                    'img_ir_np': img_ir_np,
                    'im0_rgb': im0_rgb,
                    'im0_ir': im0_ir,
                    'im0_rgb_aligned': im0_rgb_aligned,
                    'timestamp': rgb_msg.header.stamp,
                    'frame_id': rgb_msg.header.frame_id
                })
            except queue.Full:
                # 队列满，丢弃帧，不阻塞 ROS 回调
                pass

        except Exception as e:
            print(f"Error in callback: {e}")

    def inference_loop(self):
        """
        消费者：从队列取数据，GPU推理，后处理，发布
        """
        
        while not rospy.is_shutdown():
                # 设置 timeout 以便能够响应 rospy.is_shutdown()
            try:
                data = self.input_queue.get(timeout=1.0)
            except queue.Empty:
                continue  # 队列空，继续循环检查 rospy.is_shutdown()

            t1 = time_synchronized()

            img_rgb_np = data['img_rgb_np']
            img_ir_np = data['img_ir_np']
            im0_rgb = data['im0_rgb']
            im0_ir = data['im0_ir']
            im0_rgb_aligned = data['im0_rgb_aligned']
            timestamp = data['timestamp']
            frame_id = data['frame_id']

            # 1. GPU Preprocess (To Tensor)
            img_rgb = self.preprocess_img_gpu(img_rgb_np)
            img_ir = self.preprocess_img_gpu(img_ir_np)

            # 2. Inference
            with torch.no_grad():
                pred = self.model(img_rgb, img_ir, augment=self.opt.augment)[0]
                #self.model返回的是元组，[0]代表第一个元素，即预测结果，([batch, num_boxes, 5 + num_classes]),num_boxes是预测框数量，5是xywh和置信度，num_classes是类别数

            # 3. NMS
            pred = non_max_suppression(pred, self.opt.conf_thres, self.opt.iou_thres, classes=self.opt.classes, agnostic=self.opt.agnostic_nms)
            #这里的pred是一个列表，长度等于batch size，每个元素是该图像的检测结果
            # 4. Process detections
            det = pred[0]  #det是一个tensor，shape为[num_detections, 6]，每行是[x1, y1, x2, y2, conf, class]

            if len(det):
                #把检测框从imgsz大小映射回原图大小
                det[:, :4] = scale_coords(img_rgb.shape[2:], det[:, :4], im0_rgb_aligned.shape).round()
                
                # 生成掩膜
                result = generate_mask(det, im0_rgb, im0_ir, self.H_inv)
                """ cv2.imshow("Masked RGB", result["masked"][0])
                cv2.imshow("Masked IR", result["masked"][1])
                cv2.waitKey(1) """

                #-------------使用superpoint + lightglue进行特征匹配--------------
                if result is not None:
                    #用裁减后的掩膜图像提取特征并匹配
                    tensor_rgb = self.preprocess_feature_detect_image(result["masked"][0])
                    tensor_ir = self.preprocess_feature_detect_image(result["masked"][1])

                    #用原图提取特征并匹配
                    tensor_rgb = self.preprocess_feature_detect_image(im0_rgb)
                    tensor_ir = self.preprocess_feature_detect_image(im0_ir)

                    feats0 = self.extractor.extract(tensor_rgb) 
                    feats1 = self.extractor.extract(tensor_ir)
                    #feats0和feats1是字典，包含
                    # 'keypoints': 关键点坐标，形状为[1, N, 2]
                    # 'scores': 关键点分数，形状为[1, N]
                    # 'descriptors': 关键点描述子，形状为[1, N, 256]

                    #如果用的是crop图，则需要调整关键点坐标到原图位置
                    # result["crop_coords"] 是 (crop_rgb_x1, crop_rgb_y1, crop_ir_x1, crop_ir_y1)
                    """ feats0['keypoints'][..., 0] += result["crop_coords"][0]  
                    feats0['keypoints'][..., 1] += result["crop_coords"][1]
                    feats1['keypoints'][..., 0] += result["crop_coords"][2]
                    feats1['keypoints'][..., 1] += result["crop_coords"][3] """

                    matches01 = self.matcher({"image0": feats0, "image1": feats1}) #matches01是一个字典，包含matches, confidence等

                    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]] #去除张量batch维度
                    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"] 
                    #取出关键点和匹配关系，kpts0和kpts1形状为[N, 2]，matches形状为[M, 2]，M是匹配对数量，matches中每行是(kpt0_index, kpt1_index)
                    #print(len(kpts0), len(kpts1))

                    if len(matches) > 0:
                        m_kpts0 = kpts0[matches[..., 0]].cpu().numpy() #m_kpts0=([160, 241], [123, 321], ...)
                        m_kpts1 = kpts1[matches[..., 1]].cpu().numpy() #对应关键点的坐标

                        # 极线约束剔除外点
                        valid_kpts0, valid_kpts1 = correct_points(self.F, m_kpts0, m_kpts1, threshold=1.0)
                        #print(len(valid_kpts0), len(valid_kpts1))
                        """ vis_img = draw_matches(im0_rgb, im0_ir, valid_kpts0, valid_kpts1)
                        cv2.imshow("Matched Features", vis_img)
                        cv2.waitKey(1) """
                    
                    # 确保 valid_kpts0 和 valid_kpts1 在没有进入 if 分支时也被初始化，防止后面引用报错
                    if 'valid_kpts0' not in locals():
                         valid_kpts0 = np.empty((0, 2))
                         valid_kpts1 = np.empty((0, 2))

                    #print(f"Matched {len(m_kpts0)} keypoints between RGB and IR images.")
                    # 可视化匹配结果

                    # 发布可视化图像

                # ——-----------------匹配结束-----------------

                for *xyxy, conf, cls in reversed(det): # *xyxy表示前四个元素打包成列表
                    c = int(cls)  #c是类别索引
                    class_name = self.names[c]  #获取类别名称
                    x1, y1, x2, y2 =  [float(x) for x in xyxy]
                    rgb_xyxy, rgb_center = realign_boxes(xyxy, im0_rgb, self.H_inv)
                    
                    # IR 中心点
                    ir_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    
                    # --- 策略：找到离检测框中心最近的特征点 ---
                    object_3d_pos = None
                    
                    if len(valid_kpts1) > 0:
                        # 1. 筛选落在当前 RGB 框内的特征点
                        # 向量化操作筛选索引
                        mask_in_box = (valid_kpts1[:, 0] >= x1) & (valid_kpts1[:, 0] < x2) & \
                                      (valid_kpts1[:, 1] >= y1) & (valid_kpts1[:, 1] < y2)
                        
                        box_kpts0 = valid_kpts0[mask_in_box]
                        box_kpts1 = valid_kpts1[mask_in_box]

                        if len(box_kpts1) > 0:
                            # 2. 计算所有有效特征点的三维坐标
                            points_3d_list = []
                            valid_indices = []

                            for i in range(len(box_kpts0)):
                                pos_3d = triangulate_refine_gauss_newton(
                                    self.K_rgb, self.K_ir, self.R, self.t,
                                    box_kpts0[i], box_kpts1[i], iters=10
                                )
                                if pos_3d[2] > 0:  #仅保留正深度点
                                    points_3d_list.append(pos_3d)
                                    valid_indices.append(i)
                            
                            if len(points_3d_list) > 0:
                                points_3d_arr = np.array(points_3d_list)

                                # 3. 找到深度值(z)的中位数对应的特征点
                                # argsort 返回从小到大的索引，取中间那个
                                zs = points_3d_arr[:, 2]
                                sorted_indices = np.argsort(zs)
                                median_idx = sorted_indices[len(zs) // 2]
                                
                                object_3d_pos = points_3d_arr[median_idx]

                                original_idx = valid_indices[median_idx]
                                best_pt_rgb = box_kpts0[original_idx]

                                cv2.circle(im0_rgb, (int(best_pt_rgb[0]), int(best_pt_rgb[1])), radius=5, color=(0, 255, 255), thickness=-1)
                                vis_img = draw_matches(im0_rgb, im0_ir, box_kpts0, box_kpts1)
                                cv2.imshow("Best Matched Feature", vis_img)
                                cv2.waitKey(1)
                                
                                # 在图上画出选中的那个特征点（黄色）
                                
                            
                            else:
                                object_3d_pos = [0.0, 0.0, 0.0]
                        else:
                            object_3d_pos = [0.0, 0.0, 0.0]
                    else:
                        object_3d_pos = [0.0, 0.0, 0.0]

                    label = f'{self.names[int(cls)]} {conf:.2f} x:{object_3d_pos[0]:.1f} y:{object_3d_pos[1]:.1f} z:{object_3d_pos[2]:.1f}m'

                    plot_one_box(rgb_xyxy, im0_rgb, label=label, color=colors(int(cls), True), line_thickness=3)
                    plot_one_box(xyxy, im0_ir, label=label, color=colors(int(cls), True), line_thickness=3)
                    cv2.circle(im0_rgb, (int(rgb_center[0]), int(rgb_center[1])), radius=3, color=(0, 0, 255), thickness=-1)
                    print(object_3d_pos)
            cv2.imshow("Detections RGB", im0_rgb)
            cv2.waitKey(1)
            # 5. Publish
            self.pub_rgb.publish(cv2_to_imgmsg(im0_rgb, timestamp, frame_id))
            self.pub_ir.publish(cv2_to_imgmsg(im0_ir, timestamp, frame_id))
            #print(im0_rgb.shape, im0_ir.shape)
            
            self.count += 1
            if self.count % 10 == 0:
                t2 = time_synchronized()
                print(f"Processed {self.count} frames. Average FPS: {1 / (t2 - t1):.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/mydata_1_train/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')  #推理时数据增强，这里不可用

    opt = parser.parse_args()

    print("torch version:", torch.__version__)
    print("torch cuda version:", torch.version.cuda)
    print("is_available:", torch.cuda.is_available())
    print("=" * 50)
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"cuDNN 是否可用: {torch.backends.cudnn.enabled}")
    print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
    print("=" * 50)
    
    try:
        detector = DualStreamDetector(opt)
        rate = rospy.Rate(10)  # 10hz
        while not rospy.is_shutdown():
            rate.sleep()
        #rospy.spin()
    except rospy.ROSInterruptException:
        pass