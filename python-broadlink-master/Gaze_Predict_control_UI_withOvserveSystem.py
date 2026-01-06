"""
这个版本的代码能够使用注视点选中UI界面的功能块儿，且UI能全屏显示，添加了空调控制功能，用于实验阶段验证理论的可靠性
增加了多级菜单功能，用于控制智慧家居，第一级菜单为选择智慧家居，二级菜单为具体控制选项。
监视功能：在主显示器上添加2×2布局的监视窗口，显示摄像头画面、第二显示器画面、当前执行功能和操作设备。
"""

import time
import cv2
import numpy as np
import mediapipe as mp
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt, QPoint, QRect
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QGridLayout
from PyQt5.QtGui import QPainter, QColor, QPen, QFont, QImage, QPixmap
import sys
import broadlink
import json
import os
from getpass import getpass
import socket

# 获取本地 IP 地址
def get_local_ip():
    """获取本机 IP 地址"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# 连接 Broadlink 设备
def connect_device():
    local_ip = get_local_ip()
    print(f"本地 IP: {local_ip}")

    try:
        print("尝试自动发现设备...")
        devices = broadlink.discover(
            timeout=50,
            local_ip_address=local_ip,
            discover_ip_address="255.255.255.255"
        )
        if devices:
            device = devices[0]
            device.auth()
            return device
        else:
            print("未发现设备")
    except Exception as e:
        print(f"自动发现失败: {e}")

    try:
        print("尝试手动连接设备...")
        device = broadlink.hello(
            host=("172.20.10.2", 80),
            mac=bytearray.fromhex("e8165606ed15")
        )
        device.auth()
        return device
    except Exception as e:
        print(f"手动连接失败: {e}")

    print("所有连接方式均失败")
    return None

# 检查并创建 ir_codes.json 文件
def ensure_ir_codes_file():
    if not os.path.exists("ir_codes.json"):
        with open("ir_codes.json", "w") as f:
            json.dump({}, f)
        print("已创建空的 ir_codes.json 文件")

# 连接设备
device = connect_device()
if not device:
    exit("无法连接设备")
print(f"已连接: {device.type} @ {device.host}")

# GazeVisualizer 类：负责眼动追踪和校准，新增摄像头画面信号和特征绘制
class GazeVisualizer(QObject):
    gaze_update = pyqtSignal(tuple)
    camera_frame_update = pyqtSignal(np.ndarray)  # 新增信号，用于传递摄像头画面

    def __init__(self):
        super().__init__()
        screen_info = self.get_screen_resolution()
        self.screen_w = screen_info['width']
        self.screen_h = screen_info['height']
        # 获取第二显示器的信息
        screens = QApplication.screens()
        self.second_screen = screens[1] if len(screens) > 1 else screens[0]
        self.second_screen_x = self.second_screen.geometry().x()
        self.second_screen_y = self.second_screen.geometry().y()

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
        self.LEFT_EYE_INNER = 362
        self.LEFT_EYE_OUTER = 263
        self.RIGHT_EYE_INNER = 133
        self.RIGHT_EYE_OUTER = 33

        self.camera_window = "Camera Preview"
        self.calibration_window = "Calibration Points"

        self.target_points = self.generate_target_points()
        self.target_radius = 30
        self.target_color = (0, 255, 0)
        self.gaze_radius = 20
        self.gaze_color = (0, 0, 255)
        self.kalman = self.init_kalman_filter()
        self.consecutive_failures = 0

        self.calibration_data = {
            'calibrated': False,
            'features': [],
            'screen_points': [],
            'poly_coeffs_x': None,
            'poly_coeffs_y': None
        }

        self.webcam = None

    def get_screen_resolution(self):
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return {'width': width, 'height': height}

    # 初始化卡尔曼滤波器
    def init_kalman_filter(self):
        # 创建一个4维状态空间、2维测量空间的卡尔曼滤波器
        kalman = cv2.KalmanFilter(4, 2)
        # 设置测量矩阵，将状态空间的前两个维度映射到测量空间
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # 设置状态转移矩阵，描述状态之间的关系
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        # 设置过程噪声协方差矩阵，表示状态预测的不确定性
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        # 设置初始状态后验估计值，初始位置为屏幕中心，初始速度为0
        kalman.statePost = np.array([self.screen_w // 2, self.screen_h // 2, 0, 0], dtype=np.float32)
        # 返回初始化的卡尔曼滤波器
        return kalman


    def generate_target_points(self):
        points = []
        x_pos = [int(self.screen_w * 0.1), int(self.screen_w * 0.5), int(self.screen_w * 0.9)]
        y_pos = [int(self.screen_h * 0.1), int(self.screen_h * 0.5), int(self.screen_h * 0.9)]
        for x in x_pos:
            for y in y_pos:
                points.append((x, y))
        return points

    def create_fullscreen_canvas(self):
        return np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

    def draw_calibration_point(self, canvas, x, y, highlight=False):
        color = (0, 255, 255) if highlight else self.target_color
        cv2.circle(canvas, (x, y), self.target_radius + 5, (255, 255, 255), 2)
        cv2.circle(canvas, (x, y), self.target_radius, color, -1)
        return canvas

    def apply_clahe(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def get_eye_features(self, frame):
        h, w = frame.shape[0], frame.shape[1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            def get_coord(index):
                lm = face_landmarks.landmark[index]
                return (lm.x * w, lm.y * h)

            left_pupil = get_coord(self.LEFT_IRIS[0])
            right_pupil = get_coord(self.RIGHT_IRIS[0])
            left_inner = get_coord(self.LEFT_EYE_INNER)
            left_outer = get_coord(self.LEFT_EYE_OUTER)
            right_inner = get_coord(self.RIGHT_EYE_INNER)
            right_outer = get_coord(self.RIGHT_EYE_OUTER)
            eye_width = left_outer[0] - left_inner[0]
            if eye_width == 0:
                return None
            left_v = (
                (left_pupil[0] - left_inner[0]) / eye_width,
                (left_pupil[1] - left_inner[1]) / eye_width
            )
            right_v = (
                (right_pupil[0] - right_inner[0]) / eye_width,
                (right_pupil[1] - right_inner[1]) / eye_width
            )
            avg_v = (
                (left_v[0] + right_v[0]) / 2,
                (left_v[1] + right_v[1]) / 2
            )
            return avg_v
        return None

    def predict_gaze(self, v_x, v_y):
        if not self.calibration_data['calibrated']:
            return (self.screen_w // 2, self.screen_h // 2)
        try:
            features = [1, v_x, v_y, v_x * v_y, v_x ** 2, v_y ** 2]
            pred_x = np.dot(features, self.calibration_data['poly_coeffs_x'])
            pred_y = np.dot(features, self.calibration_data['poly_coeffs_y'])
            measurement = np.array([[np.float32(pred_x)], [np.float32(pred_y)]])
            self.kalman.correct(measurement)
            predicted = self.kalman.predict()
            return (
                int(np.clip(predicted[0], 0, self.screen_w)),
                int(np.clip(predicted[1], 0, self.screen_h))
            )
        except:
            return (self.screen_w // 2, self.screen_h // 2)

    def print_calibration_error(self):
        errors = []
        for (vx, vy), (true_x, true_y) in zip(
                self.calibration_data['features'],
                self.calibration_data['screen_points']
        ):
            pred_x = np.dot([1, vx, vy, vx * vy, vx ** 2, vy ** 2],
                            self.calibration_data['poly_coeffs_x'])
            pred_y = np.dot([1, vx, vy, vx * vy, vx ** 2, vy ** 2],
                            self.calibration_data['poly_coeffs_y'])
            error = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
            errors.append(error)
        print(f"平均误差: {np.mean(errors):.1f} 像素")
        print(f"最大误差: {np.max(errors):.1f} 像素")

    def calibrate(self, webcam):
        self.webcam = webcam
        print("请按顺序注视闪烁的黄色目标点")
        cv2.namedWindow(self.calibration_window, cv2.WINDOW_NORMAL)
        # 将窗口移动到第二显示器并设置为全屏
        cv2.moveWindow(self.calibration_window, self.second_screen_x, self.second_screen_y)
        cv2.setWindowProperty(self.calibration_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        try:
            for idx, (scr_x, scr_y) in enumerate(self.target_points):
                for _ in range(3):
                    canvas = self.create_fullscreen_canvas()
                    self.draw_calibration_point(canvas, scr_x, scr_y, highlight=True)
                    cv2.imshow(self.calibration_window, canvas)
                    cv2.waitKey(300)
                    canvas = self.create_fullscreen_canvas()
                    cv2.imshow(self.calibration_window, canvas)
                    cv2.waitKey(300)

                canvas = self.create_fullscreen_canvas()
                self.draw_calibration_point(canvas, scr_x, scr_y, highlight=True)
                cv2.imshow(self.calibration_window, canvas)
                cv2.waitKey(500)

                samples = []
                for _ in range(50):
                    ret, frame = webcam.read()
                    if not ret:
                        print("错误：无法读取摄像头帧")
                        continue
                    frame = self.apply_clahe(frame)
                    avg_v = self.get_eye_features(frame)
                    cv2.imshow(self.calibration_window, canvas)
                    cv2.imshow(self.camera_window, frame)
                    cv2.waitKey(10)
                    if avg_v:
                        samples.append(avg_v)

                if samples:
                    samples = np.array(samples)
                    mean = np.mean(samples, axis=0)
                    std = np.std(samples, axis=0)
                    valid = np.all(np.abs(samples - mean) < 3 * std, axis=1)
                    filtered_samples = samples[valid]
                    if len(filtered_samples) > 10:
                        avg_v = np.mean(filtered_samples, axis=0)
                        self.calibration_data['features'].append(avg_v)
                        self.calibration_data['screen_points'].append([scr_x, scr_y])

                canvas = self.create_fullscreen_canvas()
                cv2.imshow(self.calibration_window, canvas)
                cv2.waitKey(300)

            X = np.array([[1, vx, vy, vx * vy, vx ** 2, vy ** 2]
                          for vx, vy in self.calibration_data['features']])
            self.calibration_data['poly_coeffs_x'] = np.linalg.lstsq(
                X, [p[0] for p in self.calibration_data['screen_points']], rcond=None)[0]
            self.calibration_data['poly_coeffs_y'] = np.linalg.lstsq(
                X, [p[1] for p in self.calibration_data['screen_points']], rcond=None)[0]
            self.calibration_data['calibrated'] = True
            print("校准成功！误差分析：")
            self.print_calibration_error()

        except Exception as e:
            print(f"校准失败：{str(e)}")

        finally:
            cv2.destroyWindow(self.calibration_window)
            cv2.destroyWindow(self.camera_window)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            time.sleep(1.0)
            if self.webcam is not None and self.webcam.isOpened():
                self.webcam.release()
            self.webcam = None

    def init_windows(self):
        cv2.namedWindow(self.camera_window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.camera_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(self.camera_window, 0, 0)

    def run(self):
        webcam = cv2.VideoCapture(1)
        if not webcam.isOpened():
            print("错误：无法打开摄像头")
            return

        self.init_windows()
        self.calibrate(webcam)
        cv2.destroyAllWindows()
        time.sleep(1.0)

        if webcam.isOpened():
            webcam.release()

        self.webcam = cv2.VideoCapture(1)
        if not self.webcam.isOpened():
            print("错误：无法重新打开摄像头")
            return

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gaze)
        self.timer.start(30)

    def update_gaze(self):
        if self.webcam is None or not self.webcam.isOpened():
            print("错误：webcam 未初始化或已关闭")
            return
        ret, frame = self.webcam.read()
        if not ret:
            print("错误：无法读取摄像头帧")
            return
        frame = self.apply_clahe(frame)
        avg_v = self.get_eye_features(frame)
        if avg_v:
            current_gaze = self.predict_gaze(avg_v[0], avg_v[1])
            self.consecutive_failures = 0
            self.gaze_update.emit(current_gaze)
            # 绘制瞳孔和眼角特征并发送画面
            frame_with_landmarks = self.draw_landmarks(frame)
            self.camera_frame_update.emit(frame_with_landmarks)
        else:
            self.consecutive_failures += 1
            if self.consecutive_failures > 15:
                print("Tracking Lost! Please face the camera")

    def draw_landmarks(self, frame):
        """绘制瞳孔和眼角特征"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = frame.shape[:2]
            for idx in [self.LEFT_IRIS[0], self.RIGHT_IRIS[0], self.LEFT_EYE_INNER,
                        self.LEFT_EYE_OUTER, self.RIGHT_EYE_INNER, self.RIGHT_EYE_OUTER]:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        return frame

# GazePointWidget 类：绘制注视点和进度条
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
                    self.progress[i] = min(elapsed_time / 5.0, 1.0)
                    if elapsed_time > 44:
                        self.select_functional_block(i)
                return
        self.current_block = None
        self.enter_time = None
        self.progress = [0.0] * len(self.functional_blocks)

    def select_functional_block(self, block_index):
        print(f"选中功能块 {block_index}")

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

# FunctionBlockWidget 类：绘制功能块标签
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

# MonitorWindow 类：新增监视窗口，显示2×2布局
class MonitorWindow(QWidget):
    # 初始化方法
    def __init__(self, gaze_visualizer, main_window, second_screen, parent=None):
        # 调用父类的初始化方法
        super().__init__(parent)
        # 设置窗口标题
        self.setWindowTitle("Monitor Window")
        # 设置窗口几何位置，使用主显示器的几何位置
        self.setGeometry(QApplication.screens()[0].geometry())  # 主显示器
        # 创建网格布局
        self.layout = QGridLayout(self)

        # 左上角：摄像头画面
        self.camera_label = QLabel(self)
        # 将摄像头画面添加到网格布局
        self.layout.addWidget(self.camera_label, 0, 0)

        # 右上角：第二显示器画面
        self.second_screen_label = QLabel(self)
        # 将第二显示器画面添加到网格布局
        self.layout.addWidget(self.second_screen_label, 0, 1)

        # 左下角：当前执行的功能
        self.current_action_label = QLabel("当前执行：无", self)
        # 设置当前执行功能的字体
        self.current_action_label.setFont(QFont("Arial", 14))
        # 将当前执行功能添加到网格布局
        self.layout.addWidget(self.current_action_label, 1, 0)

        # 右下角：当前操作的智慧家居
        self.current_device_label = QLabel("操控设备：无", self)
        # 设置当前操作设备的字体
        self.current_device_label.setFont(QFont("Arial", 14))
        # 将当前操作设备添加到网格布局
        self.layout.addWidget(self.current_device_label, 1, 1)

        # 连接信号
        gaze_visualizer.camera_frame_update.connect(self.update_camera_frame)
        # 连接主窗口的执行动作信号
        main_window.action_executed.connect(self.update_current_action)
        # 连接主窗口的当前设备变化信号
        main_window.current_device_changed.connect(self.update_current_device)

        # 设置定时器更新第二显示器画面
        self.second_screen = second_screen
        # 创建定时器
        self.timer = QTimer()
        # 连接定时器的超时信号到更新第二显示器画面的方法
        self.timer.timeout.connect(self.update_second_screen)
        # 启动定时器，每100ms更新一次
        self.timer.start(100)  # 每100ms更新一次


    def update_camera_frame(self, frame):
        """更新摄像头画面"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = frame_rgb.shape
        qimg = QImage(frame_rgb.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio))

    def update_second_screen(self):
        """更新第二显示器画面"""
        pixmap = self.second_screen.grabWindow(0)
        self.second_screen_label.setPixmap(pixmap.scaled(self.second_screen_label.size(), Qt.KeepAspectRatio))

    def update_current_action(self, action):
        """更新当前执行的功能"""
        self.current_action_label.setText(f"当前执行：{action}")

    def update_current_device(self, device):
        """更新当前操作的智慧家居"""
        self.current_device_label.setText(f"操控设备：{device}")

# MainWindow 类：主窗口和功能块管理，新增信号
class MainWindow(QWidget):
    action_executed = pyqtSignal(str)  # 新增信号：执行功能
    current_device_changed = pyqtSignal(str)  # 新增信号：设备变更

    def __init__(self):
        super().__init__()
        self.gaze_visualizer = GazeVisualizer()
        self.gaze_point_widget = GazePointWidget(self)
        self.gaze_point_widget.show()
        self.gaze_visualizer.gaze_update.connect(self.handle_gaze)
        self.gaze_visualizer.run()

        # 移动到第二显示器并全屏显示
        screens = QApplication.screens()
        second_screen = screens[1] if len(screens) > 1 else screens[0]
        self.move(second_screen.geometry().x(), second_screen.geometry().y())
        self.showFullScreen()

        screen = second_screen.size()
        self.menu_structure = {
            'main': ['空调', '电视', '窗帘', '电灯', '呼叫', '退出'],
            '空调': ['打开空调', '温度+', '温度-', '关闭空调', '模式', '返回'],
            '电视': ['打开电视', '音量+', '音量-', '关闭电视', '频道', '返回'],
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

        # 更新当前操作的设备
        if current_menu != 'main':
            self.current_device_changed.emit(current_menu)
        else:
            self.current_device_changed.emit("无")

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
            if self.current_hover_block.hover_duration >= 44:
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
                self.init_function_blocks(QApplication.screens()[1].size() if len(QApplication.screens()) > 1 else QApplication.primaryScreen().size())
        else:
            if label == '返回':
                self.menu_path.pop()
                self.init_function_blocks(QApplication.screens()[1].size() if len(QApplication.screens()) > 1 else QApplication.primaryScreen().size())
            else:
                print(f"执行 {current_menu} 的 {label} 功能")
                self.action_executed.emit(label)  # 发送执行功能信号
                if label == "打开空调":
                    ensure_ir_codes_file()
                    with open("ir_codes.json") as f:
                        ir_codes = json.load(f)
                    command = "aircon_on"
                    if command in ir_codes:
                        device.send_data(bytes.fromhex(ir_codes[command]))
                        print(f"已发送: {command}")
                    else:
                        print(f"未找到指令: {command}")
                elif label == "关闭空调":
                    ensure_ir_codes_file()
                    with open("ir_codes.json") as f:
                        ir_codes = json.load(f)
                    command = "aircon_off"
                    if command in ir_codes:
                        device.send_data(bytes.fromhex(ir_codes[command]))
                        print(f"已发送: {command}")
                    else:
                        print(f"未找到指令: {command}")
                elif label == "温度+":
                    ensure_ir_codes_file()
                    with open("ir_codes.json") as f:
                        ir_codes = json.load(f)
                    command = "temperature_up"
                    if command in ir_codes:
                        device.send_data(bytes.fromhex(ir_codes[command]))
                        print(f"已发送: {command}")
                    else:
                        print(f"未找到指令: {command}")
                elif label == "温度-":
                    ensure_ir_codes_file()
                    with open("ir_codes.json") as f:
                        ir_codes = json.load(f)
                    command = "temperature_down"
                    if command in ir_codes:
                        device.send_data(bytes.fromhex(ir_codes[command]))
                        print(f"已发送: {command}")
                    else:
                        print(f"未找到指令: {command}")
                elif label == "模式":
                    ensure_ir_codes_file()
                    with open("ir_codes.json") as f:
                        ir_codes = json.load(f)
                    command = "mode"
                    if command in ir_codes:
                        device.send_data(bytes.fromhex(ir_codes[command]))
                        print(f"已发送: {command}")
                    else:
                        print(f"未找到指令: {command}")
                elif label == "打开电视":
                    ensure_ir_codes_file()
                    with open("ir_codes.json") as f:
                        ir_codes = json.load(f)
                    command = "tele_on"
                    if command in ir_codes:
                        device.send_data(bytes.fromhex(ir_codes[command]))
                        print(f"已发送: {command}")
                    else:
                        print(f"未找到指令: {command}")
                elif label == "关闭电视":
                    ensure_ir_codes_file()
                    with open("ir_codes.json") as f:
                        ir_codes = json.load(f)
                    command = "tele_off"
                    if command in ir_codes:
                        device.send_data(bytes.fromhex(ir_codes[command]))
                        print(f"已发送: {command}")
                    else:
                        print(f"未找到指令: {command}")
                elif label == "音量+":
                    ensure_ir_codes_file()
                    with open("ir_codes.json") as f:
                        ir_codes = json.load(f)
                    command = "voice_up"
                    if command in ir_codes:
                        device.send_data(bytes.fromhex(ir_codes[command]))
                        print(f"已发送: {command}")
                    else:
                        print(f"未找到指令: {command}")
                elif label == "音量-":
                    ensure_ir_codes_file()
                    with open("ir_codes.json") as f:
                        ir_codes = json.load(f)
                    command = "voice_down"
                    if command in ir_codes:
                        device.send_data(bytes.fromhex(ir_codes[command]))
                        print(f"已发送: {command}")
                    else:
                        print(f"未找到指令: {command}")
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    screens = QApplication.screens()
    main_screen = screens[0]
    second_screen = screens[1] if len(screens) > 1 else main_screen

    window = MainWindow()
    monitor_window = MonitorWindow(window.gaze_visualizer, window, second_screen)
    monitor_window.show()

    sys.exit(app.exec_())