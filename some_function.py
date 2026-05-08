import cv2
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import Image
import torch

#—-----图像投影对齐-------
def align_images(img1, img2, H):
        im2_height, im2_width = img2.shape[:2]
        img1Reg = cv2.warpPerspective(img1, H, (im2_width, im2_height), borderValue=(0,0,0), 
                                 flags=cv2.INTER_CUBIC)
        return img1Reg

#-----边界框反投影-----
def realign_boxes(xyxy, img, H_inv):
    x1, y1, x2, y2 =  [float(x) for x in xyxy]
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    points = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
    rgb_points = cv2.perspectiveTransform(np.array([points]), H_inv)
    rgb_center = cv2.perspectiveTransform(np.array([[[x_center, y_center]]]), H_inv)
    rgb_center = rgb_center[0][0]
    pts = rgb_points.reshape(-1, 2) #-1表示自动计算行数，2表示每行两个元素
    rgb_x1 = int(np.min(pts[:, 0]))
    rgb_y1 = int(np.min(pts[:, 1]))
    rgb_x2 = int(np.max(pts[:, 0]))
    rgb_y2 = int(np.max(pts[:, 1]))
    h_rgb, w_rgb = img.shape[0], img.shape[1]
    rgb_x1 = max(0, min(rgb_x1, w_rgb - 1))
    rgb_y1 = max(0, min(rgb_y1, h_rgb - 1))
    rgb_x2 = max(0, min(rgb_x2, w_rgb - 1))
    rgb_y2 = max(0, min(rgb_y2, h_rgb - 1))
    rgb_xyxy = [rgb_x1, rgb_y1, rgb_x2, rgb_y2]
    return rgb_xyxy, rgb_center

#--- 以下是三维深度估计相关函数 ---
def normalize_pixel(K, pt): # 将像素坐标投影到归一化平面
    pt_h = np.array([pt[0], pt[1], 1.0])
    x = np.linalg.inv(K) @ pt_h
    return x[:2] / x[2] #返回的是归一化平面坐标，维度是2

def triangulate_svd(P1, P2, pt1, pt2):
    u1, v1 = pt1
    u2, v2 = pt2

    A = np.zeros((4, 4))
    A[0] = u1 * P1[2] - P1[0]
    A[1] = v1 * P1[2] - P1[1]
    A[2] = u2 * P2[2] - P2[0]
    A[3] = v2 * P2[2] - P2[1]

    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X /= X[3]
    return X[:3] #X是4维的，返回前三维

def reprojection_error(X, P, pt): # 计算单个点的重投影误差
    X_h = np.hstack((X, 1.0)) #齐次坐标
    proj = P @ X_h   # 投影到像素坐标系
    proj /= proj[2]  # 归一化
    return proj[:2] - pt

def triangulate_refine_gauss_newton(K1, K2, R, t, pt1, pt2, iters=10):

    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = np.hstack((R, t))
    pt1n = normalize_pixel(K1, pt1) #将像素坐标投影到归一化平面，维度是2
    pt2n = normalize_pixel(K2, pt2)
    X = triangulate_svd(P1, P2, pt1n, pt2n)
    P1 = K1 @ P1
    P2 = K2 @ P2
    last_r = np.zeros(4, dtype=np.float64)

    for _ in range(iters):
        r1 = reprojection_error(X, P1, pt1)
        r2 = reprojection_error(X, P2, pt2)
        r = np.hstack((r1, r2)) # 总重投影误差向量

        J = np.zeros((4, 3))
        eps = 1e-6

        for i in range(3):
            dX = np.zeros(3) #初始化三维微小扰动向量
            dX[i] = eps     #分别对X的每个分量施加微小扰动
            J[:, i] = (  #(f(X+dx)-f(X))/dx
                np.hstack((
                    reprojection_error(X + dX, P1, pt1),
                    reprojection_error(X + dX, P2, pt2)
                )) - r
            ) / eps
        #微小扰动求解J比解析解精度稍低，但速度更快
        delta = -np.linalg.lstsq(J, r, rcond=None)[0] #np.linalg.lstsq求解最小二乘问题 JTJ*delta = -JT*r
        X += delta
        delta_r = r - last_r
        err_norm = np.linalg.norm(r) #重投影误差范数
        delta_r_norm = np.linalg.norm(delta_r) #重投影误差变化量范数
        last_r_norm = np.linalg.norm(last_r) #上次重投影误差范数
        if (last_r_norm > 1e-10 and delta_r_norm / last_r_norm < 0.001) or err_norm < 0.5:
            break
        last_r = r

    return X

#-----------以下是极线约束相关函数 -----------
def skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def compute_fundamental_matrix(K1_inv, K2_inv, R, t):
    """
    计算基础矩阵 F，使得 x2.T @ F @ x1 = 0
    假设 P2 = R @ P1 + t
    """
    t_skew = skew(t.flatten())
    E = t_skew @ R  # 本质矩阵 E = [t]x R
    F = K2_inv.T @ E @ K1_inv
    return F

def correct_points(F, m_kpts0, m_kpts1, threshold=3.0):
    valid_kpts0 = []
    valid_kpts1 = []

    for i in range(len(m_kpts0)):
        pt1 = m_kpts0[i]
        pt2 = m_kpts1[i]
        
        # 1. 检查 pt1 到 pt2 的极线的距离
        # correct_rgb_point_with_epipolar_line 返回 (u_corr, v_corr, err)
        # 这里实际上我们只需要误差来进行筛选
        _, _, err1 = correct_rgb_point_with_epipolar_line(pt1, pt2, F)
        
        # 2. 检查 pt2 到 pt1 的极线的距离 (双向约束)
        # 极线 l2 = F @ pt1 (在 IR 图像上的直线)
        p1_h = np.array([pt1[0], pt1[1], 1.0])
        l2 = F @ p1_h
        a2, b2, c2 = l2
        norm2 = a2**2 + b2**2
        if norm2 < 1e-18:
            err2 = float('inf')
        else:
            err2 = abs(a2 * pt2[0] + b2 * pt2[1] + c2) / np.sqrt(norm2)

        # 仅当双向误差都小于阈值时保留
        if err1 < threshold and err2 < threshold: 
            valid_kpts0.append(pt1)
            valid_kpts1.append(pt2)
            
    #print(f"Validation: {len(valid_kpts0)}/{len(m_kpts0)} points kept. Threshold={threshold}")
    return np.array(valid_kpts0), np.array(valid_kpts1)

def correct_rgb_point_with_epipolar_line(pt_rgb, pt_ir, F):
    """
    将初始的 RGB 点投影到由 IR 点确定的极线上。
    约束方程: pt_ir.T @ F @ pt_rgb = 0
    极线方程 l = F.T @ pt_ir (在 RGB 图像上的直线 ax + by + c = 0)
    """
    u_ir, v_ir = pt_ir
    u_rgb, v_rgb = pt_rgb

    # 构造 IR 点的齐次坐标
    p_ir_h = np.array([u_ir, v_ir, 1.0])

    # 计算 RGB 图上的极线 l = [a, b, c]
    # 注意：对应关系是 p_ir.T @ F @ p_rgb = 0 
    # 所以 (p_ir.T @ F) 就是极线系数向量的转置
    l = F.T @ p_ir_h
    a, b, c = l

    # 将 pt_rgb_initial (u0, v0) 投影到直线 ax + by + c = 0 上
    # 投影公式:
    norm = a**2 + b**2
    if norm < 1e-18:
        return pt_rgb[0], pt_rgb[1], 0.0 # 避免除零，返回原点和0误差

    d = (a * u_rgb + b * v_rgb + c) / norm
    u_rgb_corr = u_rgb - a * d
    v_rgb_corr = v_rgb - b * d
    
    # 计算几何距离（像素误差）
    pixel_error = abs(a * u_rgb + b * v_rgb + c) / np.sqrt(norm)

    return u_rgb_corr, v_rgb_corr, pixel_error

#使用 NCC 在极线上搜索最佳匹配点
def find_best_match_along_epipolar_line(img_rgb, img_ir, pt_rgb_initial, pt_ir, line_coeffs, search_range=30):
    """
    在极线上搜索最佳匹配点。
    使用 NCC (归一化互相关) 比较 11x11 的 patch。
    """
    # 确保输入是灰度图或单通道
    if len(img_rgb.shape) == 3:
        gray_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    else:
        gray_rgb = img_rgb
        
    if len(img_ir.shape) == 3:
        gray_ir = cv2.cvtColor(img_ir, cv2.COLOR_BGR2GRAY)
    else:
        gray_ir = img_ir

    u_ir, v_ir = int(pt_ir[0]), int(pt_ir[1])
    u_rgb_0, v_rgb_0 = pt_rgb_initial
    
    # 极线方向向量 (-b, a)
    a, b, c = line_coeffs
    norm = np.sqrt(a*a + b*b)
    if norm < 1e-6: return pt_rgb_initial
    
    # 单位方向向量
    dx = -b / norm
    dy = a / norm

    # 提取 IR 模板 (Reference)
    radius = 5 # 11x11 window
    h, w = gray_ir.shape
    
    # 边界保护
    if u_ir - radius < 0 or u_ir + radius >= w or v_ir - radius < 0 or v_ir + radius >= h:
        return pt_rgb_initial # 无法提取模板，直接返回初始点
        
    template = gray_ir[v_ir-radius:v_ir+radius+1, u_ir-radius:u_ir+radius+1]
    
    best_score = -1.0
    best_pt = pt_rgb_initial
    
    # 在极线上滑动搜索
    # 步长 1.0 像素
    steps = np.arange(-search_range, search_range, 1.0)
    
    for s in steps:
        # 当前候选点
        curr_u = int(u_rgb_0 + s * dx)
        curr_v = int(v_rgb_0 + s * dy)
        
        # 边界检查
        if curr_u - radius < 0 or curr_u + radius >= w or curr_v - radius < 0 or curr_v + radius >= h:
            continue
            
        patch = gray_rgb[curr_v-radius:curr_v+radius+1, curr_u-radius:curr_u+radius+1]
        
        # 计算 NCC
        res = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)
        score = res[0][0]
        
        # 很多时候 IR 和 RGB 是反色的（热=白 vs 物体=暗），试试绝对值或者负相关
        # 这里假设是正相关，如果是反色，可以取abs(score)或者 -score
        if score > best_score:
            best_score = score
            best_pt = (u_rgb_0 + s * dx, v_rgb_0 + s * dy)
            
    return best_pt

#使用 NCC 在极线上搜索最佳匹配点（简化版）
def search_on_epipolar_line(img_rgb, img_ir, pt_rgb_proj, pt_ir, line_coeffs, search_range=30):
    """
    img_rgb, img_ir: 灰度图像 (必须是单通道)
    pt_rgb_proj: 垂直投影到极线上的初始 RGB 点 (float)
    pt_ir: IR 图像上的对应点 (用于提取模板)
    line_coeffs: 极线 [a, b, c]
    search_range: 搜索半径
    """
    # 1. 提取 IR 模板 (例如 11x11 的块)
    r = 5
    x_ir, y_ir = int(pt_ir[0]), int(pt_ir[1])
    # 边界检查省略...
    template = img_ir[y_ir-r:y_ir+r+1, x_ir-r:x_ir+r+1]
    
    # 2. 确定搜索方向向量 (垂直于法向量 [a, b])
    a, b, c = line_coeffs
    # 归一化方向向量
    len_n = np.sqrt(a*a + b*b)
    dx = -b / len_n
    dy = a / len_n
    
    best_score = -1.0
    best_pt = pt_rgb_proj
    
    # 3. 沿极线滑动搜索
    for i in range(-search_range, search_range + 1):
        # 当前候选点坐标
        curr_x = int(pt_rgb_proj[0] + i * dx)
        curr_y = int(pt_rgb_proj[1] + i * dy)
        
        # 提取 RGB 候选块
        # 边界检查
        if curr_x - r < 0 or curr_y - r < 0 or ...: continue
        
        patch = img_rgb[curr_y-r:curr_y+r+1, curr_x-r:curr_x+r+1]
        
        # 计算相似度 (这里用 NCC - Normalized Cross Correlation)
        # 注意：RGB 和 IR 的灰度值可能反转，NCC 最好用绝对值或者处理过
        # 也可以尝试简单的 -abs(diff) 如果光照一致
        score = cv2.matchTemplate(patch, template, cv2.TM_CCOEFF_NORMED)[0][0]
        
        if score > best_score:
            best_score = score
            best_pt = (pt_rgb_proj[0] + i * dx, pt_rgb_proj[1] + i * dy)
            
    return best_pt

# 手动实现 imgmsg_to_cv2 和 cv2_to_imgmsg
def imgmsg_to_cv2(img_msg):
        """手动将 ROS Image 消息转换为 OpenCV 图像"""
        dtype = np.uint8
        n_channels = 3
        
        # 这里注意 到底要传入rgb图像还是bgr图像
        if img_msg.encoding == 'bgr8' or img_msg.encoding == 'rgb8':
            #print(f"img encoding is {img_msg.encoding}")  
            img_buf = np.frombuffer(img_msg.data, dtype=dtype)
            img = np.reshape(img_buf, (img_msg.height, img_msg.width, n_channels))
            if img_msg.encoding == 'rgb8':
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # if img_msg.encoding == 'bgr8':
            #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img.copy()
        elif img_msg.encoding == 'mono8':
            img_buf = np.frombuffer(img_msg.data, dtype=dtype)
            img = np.reshape(img_buf, (img_msg.height, img_msg.width))
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

def cv2_to_imgmsg(cv_image, timestamp=None, frame_id="camera"):
    """手动将 OpenCV 图像转换为 ROS Image 消息"""
    img_msg = Image()
    img_msg.header.stamp = timestamp if timestamp is not None else rospy.Time.now()
    img_msg.header.frame_id = frame_id
    img_msg.height = cv_image.shape[0]
    img_msg.width = cv_image.shape[1]
    img_msg.encoding = "bgr8"
    img_msg.is_bigendian = 0
    img_msg.step = cv_image.shape[1] * 3
    img_msg.data = cv_image.tobytes()
    return img_msg

#------------生成掩膜-----------
def generate_mask(det, im0_rgb, im0_ir, H_inv):
    masked_im0_rgb = np.zeros_like(im0_rgb)
    masked_im0_ir = np.zeros_like(im0_ir)

    # 注意边界检查，防止切片越界
    h_ir, w_ir = im0_ir.shape[:2]
    h_rgb, w_rgb = im0_rgb.shape[:2]

    crop_rgb_x1, crop_rgb_y1 = im0_rgb.shape[1], im0_rgb.shape[0]
    crop_rgb_x2, crop_rgb_y2 = 0, 0

    crop_ir_x1, crop_ir_y1 = im0_ir.shape[1], im0_ir.shape[0]
    crop_ir_x2, crop_ir_y2 = 0, 0

    for *xyxy, conf, cls in reversed(det): 
        x1, y1, x2, y2 = [int(x) for x in xyxy]
        rgb_xyxy, rgb_center = realign_boxes(xyxy, im0_rgb, H_inv)
        
        # 填充 IR 掩膜 (xyxy 是 IR/Aligned 坐标)
        # 使用 Clip 防止索引越界
        ir_y1, ir_y2 = max(0, y1), min(h_ir, y2)
        ir_x1, ir_x2 = max(0, x1), min(w_ir, x2)
        masked_im0_ir[ir_y1:ir_y2, ir_x1:ir_x2] = im0_ir[ir_y1:ir_y2, ir_x1:ir_x2]

        # 填充 RGB 掩膜 (rgb_xyxy 是映射回原图的坐标)
        rx1, ry1, rx2, ry2 = rgb_xyxy # realign_boxes 返回的是列表
        r_y1, r_y2 = max(0, ry1), min(h_rgb, ry2)
        r_x1, r_x2 = max(0, rx1), min(w_rgb, rx2)
        masked_im0_rgb[r_y1:r_y2, r_x1:r_x2] = im0_rgb[r_y1:r_y2, r_x1:r_x2]
        crop_ir_x1 = min(crop_ir_x1, ir_x1)
        crop_ir_y1 = min(crop_ir_y1, ir_y1)
        crop_ir_x2 = max(crop_ir_x2, ir_x2)
        crop_ir_y2 = max(crop_ir_y2, ir_y2)
        
        crop_rgb_x1 = min(crop_rgb_x1, r_x1)
        crop_rgb_y1 = min(crop_rgb_y1, r_y1)
        crop_rgb_x2 = max(crop_rgb_x2, r_x2)
        crop_rgb_y2 = max(crop_rgb_y2, r_y2)
    
    crop_img_ir = masked_im0_ir[crop_ir_y1:crop_ir_y2, crop_ir_x1:crop_ir_x2]
    crop_img_rgb = masked_im0_rgb[crop_rgb_y1:crop_rgb_y2, crop_rgb_x1:crop_rgb_x2]
    
    #裁减后图片可能为空
    if crop_img_rgb.size == 0 or crop_img_ir.size == 0:
        result = None
    else:
        result = {
            "masked" : (masked_im0_rgb, masked_im0_ir),
            "crop" : (crop_img_rgb, crop_img_ir),
            "crop_coords" : (crop_rgb_x1, crop_rgb_y1, crop_ir_x1, crop_ir_y1)
        }
    return result

def draw_matches(img0, img1, kpts0, kpts1, color=(0, 255, 0)):
    """
    绘制两张图片及其匹配特征点的连线。
    
    Args:
        img0: 第一张图片 (numpy array, HxW or HxWx3, BGR格式)
        img1: 第二张图片 (numpy array, HxW or HxWx3, BGR格式)
        kpts0: 第一张图片的匹配关键点 (Nx2 numpy array or torch.Tensor)
        kpts1: 第二张图片的匹配关键点 (Nx2 numpy array or torch.Tensor)
        color: 连线颜色 (B, G, R)，默认绿色
        
    Returns:
        vis: 拼接并绘制了匹配连线的图像
    """
    # 1. 确保图像是 numpy BGR 格式
    if isinstance(img0, torch.Tensor):
        img0 = img0.cpu().numpy().transpose(1, 2, 0) * 255
        img0 = img0.astype(np.uint8)
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().numpy().transpose(1, 2, 0) * 255
        img1 = img1.astype(np.uint8)
        
    if len(img0.shape) == 2:
        img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    if len(img1.shape) == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        
    # 2. 创建画布
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    vis_h = max(h0, h1)
    vis_w = w0 + w1
    vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    
    # 3. 放置图像
    vis[:h0, :w0, :] = img0
    vis[:h1, w0:w0+w1, :] = img1 # 第二张图放在右边
    
    # 4. 处理关键点 (转成 CPU numpy)
    if isinstance(kpts0, torch.Tensor):
        kpts0 = kpts0.cpu().numpy()
    if isinstance(kpts1, torch.Tensor):
        kpts1 = kpts1.cpu().numpy()
        
    # 5. 绘制连线
    for (x0, y0), (x1, y1) in zip(kpts0, kpts1):
        pt0 = (int(x0), int(y0))
        pt1 = (int(x1 + w0), int(y1)) # 偏移 x 坐标
        
        cv2.line(vis, pt0, pt1, color, 1, lineType=cv2.LINE_AA)
        cv2.circle(vis, pt0, 3, (0, 0, 255), -1) # 画红点
        cv2.circle(vis, pt1, 3, (0, 0, 255), -1)

    return vis
