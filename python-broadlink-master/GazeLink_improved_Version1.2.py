"""
Gaze_SmartHome_Depth_Compensation.py

【版本说明】
UI交互：100% 还原，包含绿色进度条与1.5秒驻留触发。
算法升级：
    1. [新增] 深度补偿 (Depth Compensation)：
       - 从 PnP 解算中提取 Z 轴距离 (Depth)。
       - 在回归模型中加入 Depth 与 Gaze Vector 的交互项 (Interaction Terms)。
       - 解决用户前后移动导致的视线映射误差（近大远小效应）。
    2. [保留] 3D 几何归一化 (抗头部转动)。
    3. [保留] Split-HSV 与 One Euro Filter (抗干扰与防抖)。
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

    NOSE = 1;
    CHIN = 152;
    L_EYE = 33;
    R_EYE = 263;
    L_MOUTH = 61;
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
#                                  GazeVisualizer (集成深度补偿)
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
            'poly_coeffs_y': None
        }

        # 特征平滑缓冲区 (增加到12帧以平滑容易抖动的深度值)
        self.feature_buffer = deque(maxlen=12)

        # 滤波器
        self.one_euro_x = None
        self.one_euro_y = None

        self.webcam = None
        self.target_radius = 30
        self.calibration_window = "Calibration"

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

    def apply_split_hsv(self, frame):
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = cv2.multiply(s, 1.5)
            s = np.clip(s, 0, 255).astype(np.uint8)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
        except:
            return frame

    def get_features(self, frame):
        """
        [算法改进] 返回特征：[vx, vy, depth]
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks: return None

        landmarks = results.multi_face_landmarks[0].landmark

        # 1. 解算 PnP 获取旋转矩阵和深度
        rmat, depth = GeometryUtils.solve_pose_and_depth(landmarks, w, h)
        if rmat is None: return None

        # 2. 3D 眼球矢量构建与归一化
        def get_3d_pt(idx):
            return np.array([landmarks[idx].x * w, landmarks[idx].y * h, landmarks[idx].z * w])

        l_iris = get_3d_pt(self.LEFT_IRIS[0])
        l_in = get_3d_pt(362);
        l_out = get_3d_pt(263)
        l_center = (l_in + l_out) / 2.0
        l_vec_raw = l_iris - l_center

        r_iris = get_3d_pt(self.RIGHT_IRIS[0])
        r_in = get_3d_pt(133);
        r_out = get_3d_pt(33)
        r_center = (r_in + r_out) / 2.0
        r_vec_raw = r_iris - r_center

        l_vec_norm = GeometryUtils.normalize_vector(l_vec_raw, rmat)
        r_vec_norm = GeometryUtils.normalize_vector(r_vec_raw, rmat)
        avg_vec = (l_vec_norm + r_vec_norm) / 2.0

        eye_width_px = np.linalg.norm(l_out - l_in)
        feature_x = avg_vec[0] / eye_width_px
        feature_y = avg_vec[1] / eye_width_px

        # 3. 将 [vx, vy, depth] 放入缓冲区平滑
        # 注意：depth 是一个较大的数值（如 500-1000），可以考虑归一化，但这里直接用线性回归也能处理
        self.feature_buffer.append(np.array([feature_x, feature_y, depth]))
        return np.mean(self.feature_buffer, axis=0)

    def predict_gaze(self, features):
        if not self.calibration_data['calibrated']:
            return (self.screen_w // 2, self.screen_h // 2)

        # 特征：vx (角度), vy (角度), z (距离)
        vx, vy, z = features

        # [算法改进] 构建多项式特征
        # 核心交互项：vx * z 和 vy * z
        # 物理意义：屏幕位移 ≈ 距离 * 角度
        inputs = np.array([1, vx, vy, z, vx * z, vy * z, vx * vy])

        try:
            pred_x = np.dot(inputs, self.calibration_data['poly_coeffs_x'])
            pred_y = np.dot(inputs, self.calibration_data['poly_coeffs_y'])

            curr_time = time.time()
            if self.one_euro_x is None:
                self.one_euro_x = OneEuroFilter(curr_time, pred_x, min_cutoff=0.01, beta=0.001)
                self.one_euro_y = OneEuroFilter(curr_time, pred_y, min_cutoff=0.01, beta=0.001)
                sx, sy = pred_x, pred_y
            else:
                sx = self.one_euro_x(curr_time, pred_x)
                sy = self.one_euro_y(curr_time, pred_y)

            return (int(np.clip(sx, 0, self.screen_w)), int(np.clip(sy, 0, self.screen_h)))
        except:
            return (self.screen_w // 2, self.screen_h // 2)

    def calibrate(self, webcam):
        self.webcam = webcam
        cv2.namedWindow(self.calibration_window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.calibration_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        print("开始校准：请保持前后距离适中（约50cm），注视屏幕绿点。")
        try:
            for tx, ty in self.target_points:
                img = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                cv2.circle(img, (tx, ty), 20, (0, 255, 0), -1)
                cv2.imshow(self.calibration_window, img)
                cv2.waitKey(1000)

                samples = []
                for _ in range(30):
                    ret, frame = webcam.read()
                    if not ret: break
                    frame = self.apply_split_hsv(frame)
                    feat = self.get_features(frame)
                    if feat is not None: samples.append(feat)
                    cv2.waitKey(10)

                if samples:
                    median_feat = np.median(samples, axis=0)
                    self.calibration_data['features'].append(median_feat)
                    self.calibration_data['screen_points'].append([tx, ty])

            # 训练模型：输入特征现在包含深度
            X = []
            for f in self.calibration_data['features']:
                vx, vy, z = f
                # 必须与 predict_gaze 中的 inputs 结构一致
                X.append([1, vx, vy, z, vx * z, vy * z, vx * vy])

            X = np.array(X)
            Tx = [p[0] for p in self.calibration_data['screen_points']]
            Ty = [p[1] for p in self.calibration_data['screen_points']]

            self.calibration_data['poly_coeffs_x'] = np.linalg.lstsq(X, Tx, rcond=None)[0]
            self.calibration_data['poly_coeffs_y'] = np.linalg.lstsq(X, Ty, rcond=None)[0]
            self.calibration_data['calibrated'] = True
            print("校准完成！深度补偿算法已启用。")

        finally:
            cv2.destroyAllWindows()

    def run(self):
        cap = cv2.VideoCapture(0)
        self.calibrate(cap)
        self.webcam = cap
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gaze)
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
#                                  UI 部分 (100% 还原自 version1.3_best.py)
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