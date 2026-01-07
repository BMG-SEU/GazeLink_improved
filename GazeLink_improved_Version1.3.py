"""
Gaze_SmartHome_Depth_Compensation_v1.3.py
【版本说明】
UI交互与逻辑：保持原版 v1.2 不变。
算法升级 (v1.3)：
    1. [新增] 抗眩光瞳孔精确定位 (Anti-Glare Pupil Refinement)：
       - 在 ROI 区域内利用加权质心法 (Weighted Centroid) 重新计算瞳孔中心。
       - 引入光斑剔除机制 (Masking)，自动忽略亮度 > 240 的反光区域。
       - 显著提升在强光、眼镜反光环境下的稳定性。
       - ！！！注意：此算法仅在检测到“强反光”时启用，否则直接信任 MediaPipe 的结果。！！！
    2. [增强] 鲁棒特征构建：
       - 使用精细化后的二维坐标结合原始深度信息构建 3D 向量。
       - 提升抗头部位移 (Head Pose Invariance) 的效果。
    3. [保留] 原有的深度补偿、PnP 解算与 One Euro Filter 平滑设置。
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt, QPoint, QRect
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
import sys
import broadlink
import json
import os
import socket
import math
from collections import deque


# ============================================================================================
#                                  Broadlink 连接逻辑 (原样保留)
# ============================================================================================

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def connect_device():
    local_ip = get_local_ip()
    print(f"本地 IP: {local_ip}")
    try:
        devices = broadlink.discover(timeout=2, local_ip_address=local_ip)
        if devices:
            device = devices[0]
            device.auth()
            print(f"已连接设备: {device.type}")
            return device
    except Exception as e:
        print(f"自动发现失败: {e}")
    print("未连接设备，进入模拟模式")
    return None


def ensure_ir_codes_file():
    if not os.path.exists("ir_codes.json"):
        with open("ir_codes.json", "w") as f:
            json.dump({}, f)


device = connect_device()
ensure_ir_codes_file()


# ============================================================================================
#                                  算法核心：几何工具与滤波器
# ============================================================================================

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=0.01, beta=0.001, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = float(x0)
        self.dx_prev = 0.0
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e, cutoff):
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a, x, x_prev):
        return a * x + (1 - a) * x_prev

    def __call__(self, t, x):
        t_e = t - self.t_prev
        if t_e <= 0: return self.x_prev
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class GeometryUtils:
    # 3D人脸标准模型 (单位：毫米，大概估算)
    FACE_3D_MODEL = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ], dtype=np.float64)

    NOSE = 1
    CHIN = 152
    L_EYE = 33
    R_EYE = 263
    L_MOUTH = 61
    R_MOUTH = 291

    @staticmethod
    def solve_pose_and_depth(landmarks, img_w, img_h):
        """
        [算法改进] 同时解算旋转矩阵 (Rotation) 和 平移向量 (Translation)
        tvec[2] 即为深度信息 (Depth)
        """
        image_points = np.array([
            (landmarks[GeometryUtils.NOSE].x * img_w, landmarks[GeometryUtils.NOSE].y * img_h),
            (landmarks[GeometryUtils.CHIN].x * img_w, landmarks[GeometryUtils.CHIN].y * img_h),
            (landmarks[GeometryUtils.L_EYE].x * img_w, landmarks[GeometryUtils.L_EYE].y * img_h),
            (landmarks[GeometryUtils.R_EYE].x * img_w, landmarks[GeometryUtils.R_EYE].y * img_h),
            (landmarks[GeometryUtils.L_MOUTH].x * img_w, landmarks[GeometryUtils.L_MOUTH].y * img_h),
            (landmarks[GeometryUtils.R_MOUTH].x * img_w, landmarks[GeometryUtils.R_MOUTH].y * img_h)
        ], dtype=np.float64)

        focal_length = img_w
        center = (img_w / 2, img_h / 2)
        camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                                 dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        # 使用 ITERATIVE 方法进行求解
        success, rvec, tvec = cv2.solvePnP(GeometryUtils.FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs)
        if not success: return None, None

        rmat, _ = cv2.Rodrigues(rvec)

        # tvec[2] 是 Z 轴距离（深度），虽然单位是相对的，但与距离成线性关系
        depth = tvec[2][0]
        return rmat, depth

    @staticmethod
    def normalize_vector(vector_3d, rotation_matrix):
        """矢量归一化：消除头部旋转影响"""
        inv_rotation = rotation_matrix.T
        return np.dot(inv_rotation, vector_3d)


# ============================================================================================
#                                  GazeVisualizer (集成抗眩光与深度补偿)
# ============================================================================================

class GazeVisualizer(QObject):
    gaze_update = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        # 获取屏幕分辨率信息（宽高）
        screen_info = self.get_screen_resolution()
        self.screen_w = screen_info['width']
        self.screen_h = screen_info['height']

        # 初始化MediaPipe面部网格检测模块
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,  # 视频模式（非静态图像）
            max_num_faces=1,          # 最多检测1张人脸
            refine_landmarks=True,    # 启用精细化眼睑和虹膜关键点
            min_detection_confidence=0.5,  # 最小检测置信度
            min_tracking_confidence=0.5    # 最小追踪置信度
        )

        # MediaPipe面部关键点索引定义（左右虹膜中心点）
        self.LEFT_IRIS = [468]       # 左眼虹膜中心关键点ID
        self.RIGHT_IRIS = [473]      # 右眼虹膜中心关键点ID

        # 生成校准用的屏幕目标点（3x3网格）
        self.target_points = self.generate_target_points()

        # 校准数据存储结构
        self.calibration_data = {
            'calibrated': False,            # 是否完成校准
            'features': [],                 # 存储校准过程中的特征向量
            'screen_points': [],            # 存储对应的屏幕坐标点
            'poly_coeffs_x': None,          # X轴多项式回归系数
            'poly_coeffs_y': None           # Y轴多项式回归系数
        }

        # 特征平滑缓冲区（保持原设置：12帧移动平均）
        self.feature_buffer = deque(maxlen=12)

        # One Euro滤波器初始化（用于输出坐标平滑）
        self.one_euro_x = None
        self.one_euro_y = None

        # 摄像头对象（在calibrate方法中初始化）
        self.webcam = None
        # 校准窗口半径参数
        self.target_radius = 30
        # 校准窗口名称
        self.calibration_window = "Calibration"


    def get_screen_resolution(self):
        """
        获取当前屏幕的分辨率信息
        实现原理：
        1. 使用tkinter创建临时窗口对象
        2. 通过winfo_screenwidth/winfo_screenheight获取真实分辨率
        3. 隐藏窗口避免视觉干扰
        4. 返回包含宽高的字典结构
        返回值:
            dict: 屏幕分辨率数据字典，包含以下键值对：
                  - width: 屏幕宽度（像素）
                  - height: 屏幕高度（像素）
        """
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return {'width': w, 'height': h}

    def generate_target_points(self):
        """
        生成用于眼动仪校准的屏幕目标点坐标集合
        实现原理：
        1. 在屏幕宽高方向各取3个位置（20%、50%、80%）
        2. 通过笛卡尔积生成3x3=9个校准点
        3. 返回格式为(x, y)的坐标元组列表
        参数:
            无显式参数（依赖实例属性 self.screen_w / self.screen_h）
        返回值:
            List[Tuple[int, int]]: 屏幕坐标点列表，每个元素为(x, y)像素坐标
        """
        points = []
        # 在水平方向取三个位置：左1/5、中间、右4/5
        x_pos = [int(self.screen_w * 0.2), int(self.screen_w * 0.5), int(self.screen_w * 0.8)]
        # 在垂直方向取三个位置：上1/5、中间、下4/5
        y_pos = [int(self.screen_h * 0.2), int(self.screen_h * 0.5), int(self.screen_h * 0.8)]

        # 生成所有网格点的笛卡尔积组合
        for x in x_pos:
            for y in y_pos:
                # 将坐标元组添加到结果列表
                points.append((x, y))
        return points

    def apply_split_hsv(self, frame):
        """
        对输入图像进行HSV颜色空间增强处理，提升视觉效果以辅助特征提取
        参数:
            frame (numpy.ndarray): 输入的BGR格式图像帧
        返回:
            numpy.ndarray: 处理后的BGR格式图像帧
            - 若处理成功返回增强后的图像
            - 若处理失败返回原始图像
        """
        try:
            # 将BGR图像转换为HSV颜色空间
            # HSV更有利于分离亮度和色彩信息
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # 分离HSV三个通道
            h, s, v = cv2.split(hsv)

            # 饱和度增强：将饱和度通道值乘以1.5倍
            # 使图像色彩更鲜艳，提升特征对比度
            s = cv2.multiply(s, 1.5)

            # 饱和度值裁剪到有效范围[0,255]并转换为8位无符号整型
            s = np.clip(s, 0, 255).astype(np.uint8)

            # 创建CLAHE对象用于对比度增强
            # clipLimit=2.0限制对比度增强的强度
            # tileGridSize=(8,8)定义处理网格大小
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            # 对明度通道应用CLAHE增强
            # 改善图像局部对比度，尤其适用于低光环境
            v = clahe.apply(v)

            # 合并处理后的通道并转换回BGR颜色空间
            return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        except:
            # 出现异常时返回原始图像帧
            return frame


    def refine_pupil_center(self, frame, rough_cx, rough_cy, radius=10):
        """
        [算法升级] 混合模式：
        只有检测到“强反光”时，才启用质心修正算法。
        否则直接返回 rough_cx, rough_cy (即 MediaPipe 的原始结果)，保证无光斑时的最高精度。
        参数:
            frame (numpy.ndarray): 输入的BGR格式图像帧
            rough_cx (int): 原始粗略瞳孔中心的x坐标
            rough_cy (int): 原始粗略瞳孔中心的y坐标
            radius (int): ROI区域的半径（默认10像素）
        返回:
            tuple: 精炼后的瞳孔中心坐标(x, y)
        """
        h, w = frame.shape[:2]
        x1 = max(0, rough_cx - radius)
        y1 = max(0, rough_cy - radius)
        x2 = min(w, rough_cx + radius)
        y2 = min(h, rough_cy + radius)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0: return rough_cx, rough_cy
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # --- [关键判断] 检测是否有强光斑 ---
        # 获取 ROI 区域的最亮像素值
        max_val = np.max(gray_roi)
        # 阈值判定：如果最亮像素低于 230，说明没有明显的镜面反光
        # 此时直接信任 MediaPipe (v1.2 逻辑)，因为它处理遮挡和噪声更准
        if max_val < 230:
            return rough_cx, rough_cy
        # --- 以下是原有 v1.3 的抗眩光逻辑 (仅在有光斑时执行) ---
        # 1. 高斯模糊去噪
        gray_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        # 2. 计算权重：剔除高光
        # 使用255减去灰度值得到反向权重，高光区域（>230）置零
        weights = (255.0 - gray_roi)
        weights[gray_roi > 230] = 0  # 强力剔除光斑区域
        # 降低指数（1.2次方）减少对噪点的敏感度
        weights = np.power(weights, 1.2)
        # 3. 计算质心
        sum_w = np.sum(weights)
        if sum_w <= 1.0:
            return rough_cx, rough_cy

        grid_y, grid_x = np.indices(gray_roi.shape)
        refined_dx = np.sum(grid_x * weights) / sum_w
        refined_dy = np.sum(grid_y * weights) / sum_w

        final_x = x1 + refined_dx
        final_y = y1 + refined_dy

        # 4. 安全钳制 (Safety Clamp)：防止修正过头
        dx = final_x - rough_cx
        dy = final_y - rough_cy
        # 如果修正距离超过 5 像素，可能是被眉毛/眼角干扰了，按比例回退
        if dx * dx + dy * dy > 25:
            return rough_cx * 0.7 + final_x * 0.3, rough_cy * 0.7 + final_y * 0.3

        return final_x, final_y


    def get_features(self, frame):
        """
        [算法改进] 融合抗眩光与PnP解算的特征提取
        从摄像头帧中提取归一化后的注视方向特征向量
        参数:
            frame (numpy.ndarray): 输入的BGR格式图像帧
        返回:
            numpy.ndarray: 包含三个特征的数组 [feature_x, feature_y, depth]
            或 None（当人脸检测失败时）
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks: return None

        landmarks = results.multi_face_landmarks[0].landmark

        # 1. 解算 PnP 获取旋转矩阵和深度 (用于头部姿态补偿)
        rmat, depth = GeometryUtils.solve_pose_and_depth(landmarks, w, h)
        if rmat is None: return None

        # 2. 3D 眼球矢量构建 (集成抗眩光精炼)
        def get_refined_3d_pt(idx):
            # 获取 MediaPipe 的原始粗略位置
            raw_cx = int(landmarks[idx].x * w)
            raw_cy = int(landmarks[idx].y * h)

            # [关键改进] 使用抗眩光算法精炼 2D 坐标
            refined_cx, refined_cy = self.refine_pupil_center(frame, raw_cx, raw_cy)

            # 结合精炼后的 x, y 和原始的 z (深度) 构建 3D 点
            # 注意：z 坐标仍然使用 MediaPipe 的估计值，因为单目图像无法通过质心法改善深度
            return np.array([refined_cx, refined_cy, landmarks[idx].z * w])

        def get_raw_3d_pt(idx):
            # 获取原始3D坐标（未经过抗眩光处理）
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h, landmarks[idx].z * w])

        # 对左右眼虹膜中心进行精炼
        l_iris = get_refined_3d_pt(self.LEFT_IRIS[0])
        r_iris = get_refined_3d_pt(self.RIGHT_IRIS[0])

        # 眼角点无需精炼，直接使用 MP 结果
        l_in = get_raw_3d_pt(362)  # 左眼内眼角
        l_out = get_raw_3d_pt(263) # 左眼外眼角
        l_center = (l_in + l_out) / 2.0
        # 计算左眼矢量（虹膜中心到眼角连线的向量）
        l_vec_raw = l_iris - l_center

        r_in = get_raw_3d_pt(133)  # 右眼内眼角
        r_out = get_raw_3d_pt(33)  # 右眼外眼角
        r_center = (r_in + r_out) / 2.0
        # 计算右眼矢量
        r_vec_raw = r_iris - r_center

        # 3. 矢量归一化 (消除头部旋转)
        l_vec_norm = GeometryUtils.normalize_vector(l_vec_raw, rmat)
        r_vec_norm = GeometryUtils.normalize_vector(r_vec_raw, rmat)
        avg_vec = (l_vec_norm + r_vec_norm) / 2.0

        # 4. 归一化处理 (除以眼宽以适应前后距离变化，但这与Depth特征有部分重叠，保留以兼容原逻辑)
        eye_width_px = np.linalg.norm(l_out - l_in)  # 左眼实际像素宽度
        feature_x = avg_vec[0] / eye_width_px  # X轴方向归一化
        feature_y = avg_vec[1] / eye_width_px  # Y轴方向归一化

        # 5. 将 [vx, vy, depth] 放入缓冲区平滑
        self.feature_buffer.append(np.array([feature_x, feature_y, depth]))
        return np.mean(self.feature_buffer, axis=0)


    def predict_gaze(self, features):
        """
        根据校准数据和当前特征向量预测屏幕上的注视点坐标
        参数:
            features (np.ndarray): 由get_features方法生成的特征向量 [vx, vy, depth]
                vx: 归一化后的水平注视方向特征
                vy: 归一化后的垂直注视方向特征
                depth: 当前头部到摄像头的距离（深度值）
        返回:
            tuple: 预测的屏幕坐标 (x, y)
                x: 水平方向像素坐标
                y: 垂直方向像素坐标
        """
        if not self.calibration_data['calibrated']:
            return (self.screen_w // 2, self.screen_h // 2)

        # 特征：vx (角度), vy (角度), z (距离)
        vx, vy, z = features
        # [算法改进] 构建多项式特征 (保持原有的深度交互项)
        # 创建包含常数项和交叉项的特征矩阵：
        # 1 - 常数项（偏置）
        # vx - 水平方向特征
        # vy - 垂直方向特征
        # z - 深度特征
        # vx*z - 水平方向与深度的交互项
        # vy*z - 垂直方向与深度的交互项
        # vx*vy - 双向方向交互项
        inputs = np.array([1, vx, vy, z, vx * z, vy * z, vx * vy])

        try:
            # 使用多项式回归系数计算预测值
            pred_x = np.dot(inputs, self.calibration_data['poly_coeffs_x'])
            pred_y = np.dot(inputs, self.calibration_data['poly_coeffs_y'])

            # 获取当前时间戳用于滤波器
            curr_time = time.time()

            # 初始化OneEuro滤波器（首次调用时）使得注视点更平滑
            if self.one_euro_x is None:
                # 保持原有的平滑设置
                """
                参数调整参考
                强环境光干扰  min_cutoff=0.003  更强的低通滤波，抑制噪声
                快速头部移动  beta=0.001        提高对速度变化的敏感度
                高精度交互    min_cutoff=0.007  减少滤波延迟，提升响应速度
                眼镜反光严重  min_cutoff=0.002  增强抗噪能力，但需注意可能引入的滞后
                """
                self.one_euro_x = OneEuroFilter(curr_time, pred_x, min_cutoff=0.005, beta=0.0005)
                self.one_euro_y = OneEuroFilter(curr_time, pred_y, min_cutoff=0.005, beta=0.0005)
                sx, sy = pred_x, pred_y
            else:
                # 使用已初始化的滤波器进行平滑处理
                sx = self.one_euro_x(curr_time, pred_x)
                sy = self.one_euro_y(curr_time, pred_y)

            # 将预测值限制在屏幕有效范围内并转换为整数坐标
            return (int(np.clip(sx, 0, self.screen_w)), int(np.clip(sy, 0, self.screen_h)))
        except:
            # 处理异常情况时返回屏幕中心点
            return (self.screen_w // 2, self.screen_h // 2)


    def calibrate(self, webcam):
        """执行眼动追踪系统的九点校准流程
        1. 创建全屏校准窗口
        2. 依次显示3x3网格校准点（共9个）
        3. 在每个校准点采集30帧特征数据
        4. 使用最小二乘法训练多项式回归模型
        5. 保存校准参数启用深度补偿算法
        参数:
            webcam: 已初始化的摄像头捕获对象
        返回:
            无返回值，通过修改实例属性完成校准
        """
        self.webcam = webcam
        cv2.namedWindow(self.calibration_window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.calibration_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("开始校准：请保持前后距离适中（约50cm），注视屏幕绿点。")
        try:
            # 遍历所有校准目标点（3x3网格共9个点）
            for tx, ty in self.target_points:
                # 创建纯黑背景的全屏图像
                img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                # 绘制绿色校准圆点（半径20像素）
                cv2.circle(img, (tx, ty), 20, (0, 255, 0), -1)
                cv2.imshow(self.calibration_window, img)
                # 等待1秒让用户稳定注视目标点
                cv2.waitKey(1000)

                # 采集30帧特征数据（约1秒）
                samples = []
                for _ in range(30):
                    ret, frame = webcam.read()
                    if not ret: break
                    # 应用HSV增强处理
                    frame = self.apply_split_hsv(frame)
                    # 提取归一化特征向量
                    feat = self.get_features(frame)
                    if feat is not None: samples.append(feat)
                    # 短暂等待保证采样频率
                    cv2.waitKey(10)

                # 如果成功采集有效样本，保存中位数特征
                if samples:
                    median_feat = np.median(samples, axis=0)
                    self.calibration_data['features'].append(median_feat)
                    self.calibration_data['screen_points'].append([tx, ty])

            # 构建设计矩阵X用于多项式回归
            X = []
            for f in self.calibration_data['features']:
                vx, vy, z = f
                # 特征构造包含7个维度：
                # [常数项, vx, vy, z, vx*z, vy*z, vx*vy]
                X.append([1, vx, vy, z, vx * z, vy * z, vx * vy])

            X = np.array(X)
            # 提取所有校准点的屏幕坐标
            Tx = [p[0] for p in self.calibration_data['screen_points']]
            Ty = [p[1] for p in self.calibration_data['screen_points']]

            # 使用最小二乘法求解X系数矩阵
            self.calibration_data['poly_coeffs_x'] = np.linalg.lstsq(X, Tx, rcond=None)[0]
            self.calibration_data['poly_coeffs_y'] = np.linalg.lstsq(X, Ty, rcond=None)[0]
            self.calibration_data['calibrated'] = True
            print("校准完成！抗眩光深度补偿算法已启用。")

        finally:
            cv2.destroyAllWindows()


    def run(self):
        """启动眼动追踪流程
        1. 初始化摄像头捕获
        2. 执行校准流程
        3. 配置定时器定期更新注视点
        4. 启动定时器后即进入持续追踪状态
        注意：此方法会阻塞直到校准完成
        """
        # 创建视频捕获对象（使用默认摄像头）
        cap = cv2.VideoCapture(0)

        # 执行完整的校准流程（会阻塞直到用户完成所有校准点）
        self.calibrate(cap)

        # 保存摄像头对象供后续使用
        self.webcam = cap

        # 创建定时器用于周期性更新注视点
        self.timer = QTimer()

        # 将定时器超时信号连接到更新方法
        self.timer.timeout.connect(self.update_gaze)

        # 启动定时器（30ms/次，约33fps）
        self.timer.start(30)

    def update_gaze(self):
        if not self.webcam: return
        ret, frame = self.webcam.read()
        if not ret: return

        frame = self.apply_split_hsv(frame)
        features = self.get_features(frame)

        if features is not None:
            gaze = self.predict_gaze(features)
            self.gaze_update.emit(gaze)


# ============================================================================================
#                                  UI 部分 (原样保留)
# ============================================================================================

class GazePointWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.gaze_point = None
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(0, 0, QApplication.primaryScreen().size().width(),
                         QApplication.primaryScreen().size().height())
        self.functional_blocks = []
        self.current_block = None
        self.enter_time = None
        self.progress = []

    def set_gaze_point(self, point):
        self.gaze_point = point
        self.check_functional_block(point)
        self.update()

    def set_functional_blocks(self, blocks):
        self.functional_blocks = blocks
        self.progress = [0.0] * len(blocks)

    def check_functional_block(self, point):
        for i, block in enumerate(self.functional_blocks):
            if block.contains(point):
                if self.current_block != i:
                    self.current_block = i
                    self.enter_time = time.time()
                    self.progress[i] = 0.0
                else:
                    elapsed_time = time.time() - self.enter_time
                    self.progress[i] = min(elapsed_time / 1.5, 1.0)
                return
        self.current_block = None
        self.enter_time = None
        self.progress = [0.0] * len(self.functional_blocks)

    def paintEvent(self, event):
        painter = QPainter(self)
        for i, block in enumerate(self.functional_blocks):
            painter.setBrush(QColor(200, 200, 200, 100))
            painter.drawRect(block)
            progress_bar_height = 20
            progress_bar_rect = QRect(block.left(), block.bottom() - progress_bar_height,
                                      int(block.width() * self.progress[i]), progress_bar_height)
            painter.setBrush(QColor(0, 255, 0))
            painter.drawRect(progress_bar_rect)
        if self.gaze_point:
            painter.setBrush(QColor(255, 0, 0))
            painter.drawEllipse(self.gaze_point, 10, 10)


class FunctionBlockWidget(QWidget):
    def __init__(self, rect, label, parent=None):
        super().__init__(parent)
        self.rect = rect
        self.label = label
        self.hovered = False
        self.hover_duration = 0
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setGeometry(rect)

    def set_label(self, label):
        self.label = label
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(QColor(30, 30, 30, 150))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 15, 15)
        if self.hovered:
            pen = QPen(QColor(0, 255, 255), 4)
            painter.setPen(pen)
            painter.drawRoundedRect(2, 2, self.width() - 4, self.height() - 4, 12, 12)
        else:
            painter.setPen(Qt.NoPen)
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 16, QFont.Bold)
        painter.setFont(font)
        painter.drawText(0, 0, self.width(), self.height(), Qt.AlignCenter, self.label)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.gaze_visualizer = GazeVisualizer()
        self.gaze_point_widget = GazePointWidget(self)
        self.gaze_point_widget.show()
        self.gaze_visualizer.gaze_update.connect(self.handle_gaze)
        self.gaze_visualizer.run()
        self.showFullScreen()
        screen = QApplication.primaryScreen().size()

        self.menu_structure = {
            'main': ['空调', '电视', '窗帘', '电灯', '呼叫', '退出'],
            '空调': ['打开空调', '温度+', '温度-', '关闭空调', '模式', '返回'],
            '电视': ['开/关', '音量+', '音量-', '频道+', '频道-', '返回'],
            '窗帘': ['开', '关', '停止', '速度+', '速度-', '返回'],
            '电灯': ['开/关', '亮度+', '亮度-', '色温+', '色温-', '返回'],
            '呼叫': ['紧急联系人1', '紧急联系人2', '紧急联系人3', '返回']
        }
        self.menu_path = ['main']
        self.function_blocks = []
        self.init_function_blocks(screen)

        for block in self.function_blocks:
            block.raise_()

        self.current_hover_block = None
        self.hover_timer = QTimer()
        self.hover_timer.timeout.connect(self.check_hover_duration)
        self.hover_timer.start(100)

    def init_function_blocks(self, screen):
        for block in self.function_blocks:
            block.deleteLater()
        self.function_blocks.clear()

        current_menu = self.menu_path[-1]
        if current_menu == '呼叫':
            self.create_blocks(screen, 2, 2)
        else:
            self.create_blocks(screen, 2, 3)
        self.update_function_blocks()

    def create_blocks(self, screen, rows, cols):
        margin = 20
        spacing = 20
        block_w = (screen.width() - 2 * margin - (cols - 1) * spacing) // cols
        block_h = (screen.height() - 2 * margin - (rows - 1) * spacing) // rows
        blocks = []
        for row in range(rows):
            for col in range(cols):
                x = margin + col * (block_w + spacing)
                y = margin + row * (block_h + spacing)
                rect = QRect(x, y, block_w, block_h)
                blocks.append(rect)
        self.gaze_point_widget.set_functional_blocks(blocks)
        self.function_blocks = []
        for rect in blocks:
            block = FunctionBlockWidget(rect, "", self)
            block.setGeometry(rect)
            block.show()
            self.function_blocks.append(block)

    def update_function_blocks(self):
        current_menu = self.menu_structure[self.menu_path[-1]]
        for block, label in zip(self.function_blocks, current_menu):
            block.set_label(label)

    def handle_gaze(self, gaze_point):
        if gaze_point:
            gaze_qpoint = QPoint(int(gaze_point[0]), int(gaze_point[1]))
            self.gaze_point_widget.set_gaze_point(gaze_qpoint)

            any_hover = False
            for block in self.function_blocks:
                if block.rect.contains(gaze_qpoint):
                    block.hovered = True
                    any_hover = True
                    if self.current_hover_block != block:
                        self.current_hover_block = block
                        block.hover_duration = 0
                else:
                    block.hovered = False
                block.update()

            if not any_hover:
                self.current_hover_block = None

    def check_hover_duration(self):
        if self.current_hover_block:
            self.current_hover_block.hover_duration += 1
            if self.current_hover_block.hover_duration >= 15:
                self.activate_function(self.current_hover_block.label)
                self.current_hover_block.hover_duration = 0

    def activate_function(self, label):
        print(f"\n--- 激活功能：{label} ---")
        current_menu = self.menu_path[-1]
        if current_menu == 'main':
            if label == '退出':
                self.close()
            elif label in self.menu_structure:
                self.menu_path.append(label)
                self.init_function_blocks(QApplication.primaryScreen().size())
        else:
            if label == '返回':
                self.menu_path.pop()
                self.init_function_blocks(QApplication.primaryScreen().size())
            else:
                self.execute_cmd(label)

    def execute_cmd(self, label):
        if not device: return
        try:
            with open("ir_codes.json") as f:
                codes = json.load(f)
            key_map = {
                "打开空调": "aircon_on", "关闭空调": "aircon_off",
                "温度+": "temp_up", "温度-": "temp_down"
            }
            if label in key_map and key_map[label] in codes:
                device.send_data(bytes.fromhex(codes[key_map[label]]))
                print(f"发送成功: {label}")
            else:
                print(f"未配置指令: {label}")
        except Exception as e:
            print(f"指令错误: {e}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())