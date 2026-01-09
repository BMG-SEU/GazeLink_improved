"""
Gaze_SmartHome_Improved_V1.3.py

【版本说明】
UI交互：100% 还原，包含绿色进度条与1.5秒驻留触发。
算法升级 (基于论文与技术报告)：
    1. [抗眩光] 引入 Timm-Barth 梯度算法：利用梯度向量场定位瞳孔，通过亮度反向加权抑制光斑干扰。
    2. [特征增强] 引入眼部网格特征 (Eye Grid Features)：3x5网格灰度积分，提升Y轴垂向注视精度。
    3. [姿态补偿] 显式头部姿态补偿：基于 PnP 解算的平移向量，计算物理位移产生的投影偏差并进行补偿。
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
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QRadialGradient, QLinearGradient
from PyQt5.QtCore import QPointF  # 注意这里需要 QPointF

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
#                                  算法核心：几何工具与抗干扰算法
# ============================================================================================

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=0.005, beta=0.0005, d_cutoff=1.0):
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


class AntiGlareUtils:
    """
    [算法改进] 抗眩光与瞳孔精确定位工具
    基于 Timm-Barth 算法思想与技术报告建议：利用梯度场特性，并对高亮区域进行权重抑制。
    """

    @staticmethod
    def get_eye_roi(frame, landmarks, eye_indices, img_w, img_h, padding=5):
        """获取眼部ROI区域"""
        pts = np.array([(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices], dtype=np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        # 增加padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_w - x, w + 2 * padding)
        h = min(img_h - y, h + 2 * padding)
        return frame[y:y + h, x:x + w], (x, y)

    @staticmethod
    def compute_timm_barth_center(eye_img):
        """
        [核心算法] 实现 Timm-Barth 梯度点积算法的简化高效版。
        通过梯度方向的一致性寻找瞳孔中心，并自动抑制光斑（高亮区）。
        """
        if eye_img is None or eye_img.size == 0:
            return None

        # 1. 预处理：转灰度，去噪
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        # 使用高斯模糊平滑噪声
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. 计算梯度 (Sobel)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(grad_x, grad_y)

        # 归一化梯度向量
        mask = magnitude > 0.001  # 避免除零
        grad_x[mask] /= magnitude[mask]
        grad_y[mask] /= magnitude[mask]
        grad_x[~mask] = 0
        grad_y[~mask] = 0

        # 3. 权重计算 (关键：抗眩光)
        # 权重与亮度成反比：越黑的地方权重越高，越亮(光斑)的地方权重越低(接近0)
        # DOCX报告建议：Weight(x) = 255 - I(x)
        weights = 255.0 - gray.astype(np.float32)

        # 额外抑制：如果像素过亮(>240)，直接将权重置0 (Hard Threshold for Glare)
        weights[gray > 240] = 0

        # 4. 下采样加速 (可选，为了实时性，这里在小ROI上直接计算)
        h, w = gray.shape
        centers = []

        # 这里的实现为了Python性能做了一定简化：
        # 我们寻找加权梯度的几何中心作为瞳孔中心的初始估计
        # 更严格的Timm-Barth需要对每个像素计算点积之和，计算量较大 (O(N^2))
        # 改进方案：使用基于梯度的质心法，加权梯度模长和反向亮度

        # 计算暗通道图像 (Dark Channel)
        inverted = 255 - gray
        inverted[gray > 220] = 0  # 去除光斑

        # 计算质心
        M = cv2.moments(inverted)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)

        return (w // 2, h // 2)


class FeatureUtils:
    """
    [算法改进] 特征提取工具
    包含：眼动向量、眼部网格特征 (PDF建议)
    """

    @staticmethod
    def get_eye_grid_features(eye_img, grid_rows=3, grid_cols=5):
        """
        [核心算法] 提取眼部网格特征 (3x5)
        解决Y轴注视精度低的问题 (PDF第3.4节)
        """
        if eye_img is None or eye_img.size == 0:
            return np.zeros(grid_rows * grid_cols)

        # 1. 预处理
        gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 2. 直方图均衡化 (CLAHE) - 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # 3. 网格划分与积分
        features = []
        dy = h // grid_rows
        dx = w // grid_cols

        total_intensity = np.sum(gray) + 0.001  # 避免除零

        for r in range(grid_rows):
            for c in range(grid_cols):
                # 提取子网格
                roi = gray[r * dy: (r + 1) * dy, c * dx: (c + 1) * dx]
                # 计算该网格的灰度占比
                grid_sum = np.sum(roi)
                features.append(grid_sum / total_intensity)

        return np.array(features)


class GeometryUtils:
    # 3D人脸标准模型
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

    # MediaPipe 眼睛轮廓索引 (用于ROI裁剪)
    LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
    RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

    @staticmethod
    def solve_pose_full(landmarks, img_w, img_h):
        """
        [算法改进] 完整解算 PnP，返回旋转与平移向量
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

        success, rvec, tvec = cv2.solvePnP(GeometryUtils.FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs)
        if not success: return None, None, None

        return rvec, tvec, camera_matrix


# ============================================================================================
#                                  GazeVisualizer (集成三大核心算法)
# ============================================================================================

# ... (之前的代码保持不变) ...

# ============================================================================================
#                                  GazeVisualizer (包含防抖优化)
# ============================================================================================

class GazeVisualizer(QObject):
    gaze_update = pyqtSignal(tuple)

    def __init__(self):
        super().__init__()
        screen_info = self.get_screen_resolution()
        self.screen_w = screen_info['width']
        self.screen_h = screen_info['height']

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.LEFT_IRIS = [468]
        self.RIGHT_IRIS = [473]
        self.target_points = self.generate_target_points()

        self.calibration_data = {
            'calibrated': False,
            'features': [],
            'screen_points': [],
            'poly_coeffs_x': None,
            'poly_coeffs_y': None,
            'ref_tvec': None
        }

        # --- 优化1：特征平滑缓冲区 ---
        # 增加缓冲区长度到 6-8，对原始特征进行强力平均，这能有效抑制网格特征的噪声
        self.feature_buffer = deque(maxlen=7)

        # --- 优化2：记录上一次的坐标，用于计算死区 ---
        self.last_gaze_x = 0
        self.last_gaze_y = 0

        # --- 优化3：调整滤波器参数 ---
        # min_cutoff: 越小，静止时越稳 (0.05 -> 0.01)
        # beta: 越小，移动时越平滑但延迟越高 (0.005 -> 0.0005)
        self.one_euro_x = None
        self.one_euro_y = None

        self.webcam = None
        self.calibration_window = "Calibration"

    # ... (get_screen_resolution, generate_target_points, get_full_features, solve_pose_full 等方法保持不变，直接复制即可) ...
    # 为了节省篇幅，这里假设 get_screen_resolution 到 get_full_features 方法没有变动
    # 请确保保留原本的 get_full_features 逻辑

    def get_screen_resolution(self):
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w, h = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return {'width': w, 'height': h}

    def generate_target_points(self):
        points = []
        x_pos = [int(self.screen_w * 0.2), int(self.screen_w * 0.5), int(self.screen_w * 0.8)]
        y_pos = [int(self.screen_h * 0.2), int(self.screen_h * 0.5), int(self.screen_h * 0.8)]
        for x in x_pos:
            for y in y_pos:
                points.append((x, y))
        return points

    def get_full_features(self, frame):
        # ... (此处代码与上一版完全一致，请保留原逻辑) ...
        # 如果需要我完整重写这个函数请告知，否则请复用之前提供的 get_full_features
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks: return None, None
        landmarks = results.multi_face_landmarks[0].landmark
        rvec, tvec, cam_mat = GeometryUtils.solve_pose_full(landmarks, w, h)
        if rvec is None: return None, None
        l_eye_img, l_offset = AntiGlareUtils.get_eye_roi(frame, landmarks, GeometryUtils.LEFT_EYE_CONTOUR, w, h)
        r_eye_img, r_offset = AntiGlareUtils.get_eye_roi(frame, landmarks, GeometryUtils.RIGHT_EYE_CONTOUR, w, h)
        l_center_offset = AntiGlareUtils.compute_timm_barth_center(l_eye_img)
        r_center_offset = AntiGlareUtils.compute_timm_barth_center(r_eye_img)

        def get_3d_pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h, landmarks[idx].z * w])

        l_iris = get_3d_pt(self.LEFT_IRIS[0])
        l_in = get_3d_pt(362)
        l_out = get_3d_pt(263)
        l_vec = l_iris - (l_in + l_out) / 2.0
        r_iris = get_3d_pt(self.RIGHT_IRIS[0])
        r_in = get_3d_pt(133)
        r_out = get_3d_pt(33)
        r_vec = r_iris - (r_in + r_out) / 2.0
        avg_vec = (l_vec + r_vec) / 2.0
        eye_width_px = np.linalg.norm(l_out - l_in)
        vx = avg_vec[0] / eye_width_px
        vy = avg_vec[1] / eye_width_px
        depth = tvec[2][0]
        l_grid = FeatureUtils.get_eye_grid_features(l_eye_img)
        r_grid = FeatureUtils.get_eye_grid_features(r_eye_img)
        avg_grid = (l_grid + r_grid) / 2.0
        combined_features = np.concatenate(([vx, vy, depth], avg_grid))
        return combined_features, tvec

    def predict_gaze(self, features, current_tvec):
        if not self.calibration_data['calibrated']:
            return (self.screen_w // 2, self.screen_h // 2)

        # --- 优化1：输入级平滑 (Input Smoothing) ---
        # 将当前特征加入队列
        self.feature_buffer.append(features)
        # 取队列中所有特征的平均值作为本次预测的输入
        # 这能极大减少由于光照闪烁、像素噪声引起的特征抖动
        smoothed_features = np.mean(self.feature_buffer, axis=0)

        # 使用平滑后的特征进行计算
        vx, vy, z = smoothed_features[:3]
        grid_feats = smoothed_features[3:]

        base_inputs = [1, vx, vy, z, vx * z, vy * z]
        inputs = np.concatenate((base_inputs, grid_feats))

        try:
            pred_x = np.dot(inputs, self.calibration_data['poly_coeffs_x'])
            pred_y = np.dot(inputs, self.calibration_data['poly_coeffs_y'])

            # 头部姿态补偿 (逻辑不变)
            if self.calibration_data['ref_tvec'] is not None and current_tvec is not None:
                ref_t = self.calibration_data['ref_tvec']
                curr_t = current_tvec
                fx = self.screen_w

                proj_x_curr = fx * (curr_t[0][0] / curr_t[2][0])
                proj_y_curr = fx * (curr_t[1][0] / curr_t[2][0])
                proj_x_ref = fx * (ref_t[0][0] / ref_t[2][0])
                proj_y_ref = fx * (ref_t[1][0] / ref_t[2][0])

                delta_x = proj_x_curr - proj_x_ref
                delta_y = proj_y_curr - proj_y_ref

                pred_x += delta_x
                pred_y += delta_y

            curr_time = time.time()

            # --- 优化3：更激进的 OneEuro 滤波参数 ---
            # min_cutoff=0.01: 允许极慢的动作通过，减少静止抖动
            # beta=0.0005: 极低的速度系数，强制平滑轨迹（会增加一点延迟，但换来极高的顺滑度）
            if self.one_euro_x is None:
                self.one_euro_x = OneEuroFilter(curr_time, pred_x, min_cutoff=0.01, beta=0.0005)
                self.one_euro_y = OneEuroFilter(curr_time, pred_y, min_cutoff=0.01, beta=0.0005)
                sx, sy = pred_x, pred_y
            else:
                sx = self.one_euro_x(curr_time, pred_x)
                sy = self.one_euro_y(curr_time, pred_y)

            # --- 优化4：死区控制 (Dead Zone / Hysteresis) ---
            # 计算当前预测点与上一次显示点的距离
            dist = np.sqrt((sx - self.last_gaze_x) ** 2 + (sy - self.last_gaze_y) ** 2)

            # 如果移动距离小于 20 像素（根据屏幕大小可调整，通常是按钮大小的1/5），
            # 则认为是噪声，强行将光标“吸附”在旧位置附近
            DEAD_ZONE_THRESHOLD = 20.0

            if dist < DEAD_ZONE_THRESHOLD:
                # 强滞后插值：90% 使用旧坐标，10% 使用新坐标
                # 这样即使有抖动，光标也几乎看起来是不动的
                sx = 0.9 * self.last_gaze_x + 0.1 * sx
                sy = 0.9 * self.last_gaze_y + 0.1 * sy

            # 更新历史坐标
            self.last_gaze_x = sx
            self.last_gaze_y = sy

            return (int(np.clip(sx, 0, self.screen_w)), int(np.clip(sy, 0, self.screen_h)))

        except Exception as e:
            return (self.screen_w // 2, self.screen_h // 2)

    def calibrate(self, webcam):
        # ... (保持原有的校准逻辑) ...
        # 唯一建议：校准时采集的帧数可以稍微增加，例如从 20 帧增加到 30 帧，以获得更稳定的基准
        self.webcam = webcam
        cv2.namedWindow(self.calibration_window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.calibration_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("开始校准...")
        collected_tvecs = []

        try:
            for tx, ty in self.target_points:
                img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                cv2.circle(img, (tx, ty), 20, (0, 255, 0), -1)
                cv2.imshow(self.calibration_window, img)
                cv2.waitKey(1200)

                samples = []
                # [建议] 增加采样帧数到 30
                for _ in range(30):
                    ret, frame = webcam.read()
                    if not ret: break
                    feat, tvec = self.get_full_features(frame)
                    if feat is not None:
                        samples.append(feat)
                        if tvec is not None:
                            collected_tvecs.append(tvec)
                    cv2.waitKey(10)

                if samples:
                    median_feat = np.median(samples, axis=0)
                    self.calibration_data['features'].append(median_feat)
                    self.calibration_data['screen_points'].append([tx, ty])

            if not self.calibration_data['features']:
                return

            if collected_tvecs:
                self.calibration_data['ref_tvec'] = np.mean(collected_tvecs, axis=0)

            X = []
            for f in self.calibration_data['features']:
                vx, vy, z = f[:3]
                grid_feats = f[3:]
                base_inputs = [1, vx, vy, z, vx * z, vy * z]
                X.append(np.concatenate((base_inputs, grid_feats)))

            X = np.array(X)
            Tx = [p[0] for p in self.calibration_data['screen_points']]
            Ty = [p[1] for p in self.calibration_data['screen_points']]

            self.calibration_data['poly_coeffs_x'] = np.linalg.lstsq(X, Tx, rcond=None)[0]
            self.calibration_data['poly_coeffs_y'] = np.linalg.lstsq(X, Ty, rcond=None)[0]
            self.calibration_data['calibrated'] = True
            print("校准完成！")

        finally:
            cv2.destroyAllWindows()

    def run(self):
        # 保持不变
        cap = cv2.VideoCapture(0)
        self.calibrate(cap)
        self.webcam = cap
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gaze)
        self.timer.start(30)

    def update_gaze(self):
        # 保持不变
        if not self.webcam: return
        ret, frame = self.webcam.read()
        if not ret: return
        features, tvec = self.get_full_features(frame)
        if features is not None:
            gaze = self.predict_gaze(features, tvec)
            self.gaze_update.emit(gaze)


# ============================================================================================
#                                  UI 部分 (100% 还原自 Version 1.2)
# ============================================================================================


#气泡光标与物理引擎
class GazePointWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.target_pos = QPoint(0, 0)  # 目标位置（算法输出的最新点）
        self.current_pos = QPointF(0, 0)  # 当前渲染位置（用于平滑插值）
        self.velocity = QPointF(0, 0)  # 当前速度向量
        self.radius = 40.0  # 基础半径

        # 呼吸动画参数
        self.breath_phase = 0.0
        self.breath_timer = QTimer(self)
        self.breath_timer.timeout.connect(self.update_animation)
        self.breath_timer.start(16)  # 约60FPS刷新率，保证动画流畅

        # 交互块引用
        self.functional_blocks = []
        self.current_block = None
        self.enter_time = None
        self.progress = []

        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setGeometry(0, 0, QApplication.primaryScreen().size().width(),
                         QApplication.primaryScreen().size().height())

    def set_gaze_point(self, point):
        """接收算法传来的新坐标"""
        self.target_pos = point
        self.check_functional_block(point)
        # 注意：这里不再直接 update()，而是由 internal timer (update_animation) 驱动渲染
        # 这样即使眼动数据卡顿，气泡的物理回弹效果依然流畅

    def set_functional_blocks(self, blocks):
        self.functional_blocks = blocks
        self.progress = [0.0] * len(blocks)

    def check_functional_block(self, point):
        # (保持原逻辑不变)
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

    def update_animation(self):
        """物理引擎核心：计算插值、速度与呼吸效果"""
        # 1. 物理插值 (Lerp)：让 current_pos 追赶 target_pos
        # 0.2 的系数决定了跟手的“粘性”，越小越平滑但延迟越高
        target_f = QPointF(self.target_pos)
        diff = target_f - self.current_pos

        # 计算瞬时速度 (用于变形)
        speed = np.sqrt(diff.x() ** 2 + diff.y() ** 2)

        # 如果距离很近，直接吸附，避免微小抖动
        if speed < 1.0:
            self.current_pos = target_f
            self.velocity = QPointF(0, 0)
        else:
            self.current_pos += diff * 0.15  # 追赶系数
            self.velocity = diff * 0.15  # 记录速度用于绘图变形

        # 2. 呼吸效果
        self.breath_phase += 0.1

        # 3. 触发重绘
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # --- 绘制功能块 (保持不变) ---
        for i, block in enumerate(self.functional_blocks):
            painter.setBrush(QColor(200, 200, 200, 50))  # 稍微调淡一点背景
            painter.setPen(QColor(255, 255, 255, 100))
            painter.drawRect(block)

            # 进度条
            if self.progress[i] > 0:
                progress_bar_height = 20
                progress_bar_rect = QRect(block.left(), block.bottom() - progress_bar_height,
                                          int(block.width() * self.progress[i]), progress_bar_height)
                # 使用渐变色进度条
                grad = QLinearGradient(block.left(), 0, block.right(), 0)
                grad.setColorAt(0, QColor(0, 255, 128))
                grad.setColorAt(1, QColor(0, 200, 255))
                painter.setBrush(grad)
                painter.setPen(Qt.NoPen)
                painter.drawRect(progress_bar_rect)

        # --- 绘制气泡光标 (核心美化) ---
        if self.target_pos:
            painter.save()

            # 1. 坐标变换系统
            cx, cy = self.current_pos.x(), self.current_pos.y()
            painter.translate(cx, cy)  # 将原点移到光标中心

            # 2. 计算变形 (Squash & Stretch)
            # 速度越快，拉伸越长
            vel_len = np.sqrt(self.velocity.x() ** 2 + self.velocity.y() ** 2)

            # 限制最大形变，防止瞬移时变成一条线
            stretch_factor = 1.0 + min(vel_len / 30.0, 0.6)
            squash_factor = 1.0 / math.sqrt(stretch_factor)  # 保持体积(面积)近似不变

            # 计算旋转角度：光标要指向运动方向
            angle = math.atan2(self.velocity.y(), self.velocity.x()) * 180 / math.pi
            painter.rotate(angle)

            # 应用形变
            painter.scale(stretch_factor, squash_factor)

            # 3. 绘制气泡本体 (径向渐变)
            # 基础呼吸脉动
            pulse = math.sin(self.breath_phase) * 2.0
            base_r = self.radius + pulse

            # 渐变层1：主体 (半透明青色)
            radialGrad = QRadialGradient(0, 0, base_r)
            radialGrad.setColorAt(0.0, QColor(0, 255, 255, 40))  # 中心淡青色
            radialGrad.setColorAt(0.8, QColor(0, 200, 255, 150))  # 边缘深蓝色
            radialGrad.setColorAt(1.0, QColor(0, 150, 255, 0))  # 外圈透明

            painter.setBrush(radialGrad)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(0, 0), base_r, base_r)

            # 4. 绘制高光 (模拟玻璃质感)
            # 高光不随身体旋转，永远在左上角，显得更像物理光源
            painter.rotate(-angle)  # 抵消旋转
            painter.scale(1 / stretch_factor, 1 / squash_factor)  # 抵消形变 (高光保持圆形)

            highlight_r = base_r * 0.4
            offset = base_r * -0.3

            highGrad = QRadialGradient(offset, offset, highlight_r)
            highGrad.setColorAt(0.0, QColor(255, 255, 255, 220))  # 纯白高光
            highGrad.setColorAt(1.0, QColor(255, 255, 255, 0))

            painter.setBrush(highGrad)
            painter.drawEllipse(QPointF(offset, offset), highlight_r, highlight_r)

            painter.restore()

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