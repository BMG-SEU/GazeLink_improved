import cv2
import numpy as np
import mediapipe as mp
import time
import matplotlib.pyplot as plt

"""
实验代码：检验眼动追踪算法预测的注视点的准确度；
"""
class GazeVisualizer:
    def __init__(self):
        # 获取实际屏幕分辨率（自动适配）
        screen_info = self.get_screen_resolution()
        self.screen_w = screen_info['width']
        self.screen_h = screen_info['height']

        # 初始化MediaPipe面部网格
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 定义关键点索引（MediaPipe）
        self.LEFT_IRIS = [468]  # 左瞳孔中心
        self.RIGHT_IRIS = [473]  # 右瞳孔中心
        self.LEFT_EYE_INNER = 362  # 左眼内眼角
        self.LEFT_EYE_OUTER = 263  # 左眼外眼角
        self.RIGHT_EYE_INNER = 133  # 右眼内眼角
        self.RIGHT_EYE_OUTER = 33   # 右眼外眼角

        # 窗口系统（保持不变）
        self.camera_window = "Camera Preview"
        self.gaze_window = "Gaze Prediction"
        self.calibration_window = "Calibration Points"

        # 其他参数保持不变
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

        # 添加实验相关数据结构
        self.experiment_data = {
            'target_points': self.target_points,
            'sampled_gazes': {i: [] for i in range(9)},  # 存储每个目标点的采样注视点
            'errors': []  # 存储每个目标点的平均误差
        }
        self.colors = self.generate_colors(9)  # 为每个目标点生成唯一颜色

    def generate_colors(self, n):
        """生成n个不同的颜色"""
        cmap = plt.get_cmap('tab10')
        return [tuple(int(c * 255) for c in cmap(i)[:3]) for i in range(n)]

    def get_eye_features(self, frame):
        """使用MediaPipe获取眼部特征"""
        h, w = frame.shape[0], frame.shape[1]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        features = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]  # 取第一张人脸

            # 获取关键点坐标
            def get_coord(index):
                lm = face_landmarks.landmark[index]
                return (lm.x * w, lm.y * h)

            # 瞳孔坐标
            left_pupil = get_coord(self.LEFT_IRIS[0])
            right_pupil = get_coord(self.RIGHT_IRIS[0])

            # 眼角坐标
            left_inner = get_coord(self.LEFT_EYE_INNER)
            left_outer = get_coord(self.LEFT_EYE_OUTER)
            right_inner = get_coord(self.RIGHT_EYE_INNER)
            right_outer = get_coord(self.RIGHT_EYE_OUTER)

            # 计算归一化特征
            eye_width = left_outer[0] - left_inner[0]
            if eye_width == 0:  # 防止除零错误
                return None

            # 左眼归一化
            left_v = (
                (left_pupil[0] - left_inner[0]) / eye_width,
                (left_pupil[1] - left_inner[1]) / eye_width
            )

            # 右眼归一化
            right_v = (
                (right_pupil[0] - right_inner[0]) / eye_width,
                (right_pupil[1] - right_inner[1]) / eye_width
            )

            # 取双眼平均值
            avg_v = (
                (left_v[0] + right_v[0]) / 2,
                (left_v[1] + right_v[1]) / 2
            )

            return avg_v
        return None

    def calibrate(self, webcam):
        """修改后的校准流程"""
        print("请按顺序注视闪烁的黄色目标点")
        cv2.setWindowProperty(self.calibration_window,
                              cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)

        for idx, (scr_x, scr_y) in enumerate(self.target_points):
            # 闪烁阶段（保持原逻辑不变）
            for _ in range(3):
                canvas = self.create_fullscreen_canvas()
                self.draw_calibration_point(canvas, scr_x, scr_y, highlight=True)
                cv2.imshow(self.calibration_window, canvas)
                cv2.waitKey(300)
                canvas = self.create_fullscreen_canvas()
                cv2.imshow(self.calibration_window, canvas)
                cv2.waitKey(300)

            # 保持显示阶段
            canvas = self.create_fullscreen_canvas()
            self.draw_calibration_point(canvas, scr_x, scr_y, highlight=True)
            cv2.imshow(self.calibration_window, canvas)
            cv2.waitKey(500)

            # 数据收集（使用新特征提取方法）
            samples = []
            for _ in range(50):
                _, frame = webcam.read()
                frame = self.apply_clahe(frame)
                avg_v = self.get_eye_features(frame)

                # 显示预览
                cv2.imshow(self.calibration_window, canvas)
                cv2.imshow(self.camera_window, frame)
                cv2.waitKey(10)

                if avg_v:
                    samples.append(avg_v)

            # 过滤和处理样本（保持原逻辑）
            if samples:
                samples = np.array(samples)
                mean = np.mean(samples, axis=0)
                std = np.std(samples, axis=0)
                valid = np.all(np.abs(samples - mean) < 3*std, axis=1)
                filtered_samples = samples[valid]
                if len(filtered_samples) > 10:
                    avg_v = np.mean(filtered_samples, axis=0)
                    self.calibration_data['features'].append(avg_v)
                    self.calibration_data['screen_points'].append([scr_x, scr_y])

            # 清除当前校准点
            canvas = self.create_fullscreen_canvas()
            cv2.imshow(self.calibration_window, canvas)
            cv2.waitKey(300)

        # 关闭校准窗口并拟合（保持原逻辑）
        cv2.destroyWindow(self.calibration_window)
        try:
            X = np.array([[1, vx, vy, vx * vy, vx ** 2, vy ** 2]
                          for vx, vy in self.calibration_data['features']])
            self.calibration_data['poly_coeffs_x'] = np.linalg.lstsq(
                X, [p[0] for p in self.calibration_data['screen_points']], rcond=None)[0]
            self.calibration_data['poly_coeffs_y'] = np.linalg.lstsq(
                X, [p[1] for p in self.calibration_data['screen_points']], rcond=None)[0]
            self.calibration_data['calibrated'] = True
            print("校准成功！误差分析：")
            self.print_calibration_error()
        except:
            print("校准失败：矩阵奇异或数据不足")

    def run_accuracy_experiment(self, webcam):
        """进行准确度实验"""
        print("开始准确度实验...")
        cv2.setWindowProperty(self.gaze_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        for idx, (target_x, target_y) in enumerate(self.target_points):
            print(f"实验点 {idx+1}/9: ({target_x}, {target_y})")
            # 显示目标点
            gaze_canvas = self.create_fullscreen_canvas()
            cv2.circle(gaze_canvas, (target_x, target_y), self.target_radius, self.target_color, -1)
            cv2.putText(gaze_canvas, f"Target {idx+1}", (target_x + 50, target_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.imshow(self.gaze_window, gaze_canvas)
            cv2.waitKey(100)

            # 等待三秒钟让注视点稳定
            time.sleep(3)

            # 采样五秒，每0.5秒采样一次
            start_time = time.time()
            sample_count = 0
            while time.time() - start_time < 5:
                _, frame = webcam.read()
                frame = self.apply_clahe(frame)
                avg_v = self.get_eye_features(frame)
                if avg_v:
                    current_gaze = self.predict_gaze(avg_v[0], avg_v[1])
                    # 实时绘制注视点
                    cv2.circle(gaze_canvas, current_gaze, self.gaze_radius, self.gaze_color, -1)
                    cv2.imshow(self.gaze_window, gaze_canvas)
                    cv2.waitKey(10)

                    # 每0.5秒采样一次
                    if (time.time() - start_time) >= sample_count * 0.5:
                        self.experiment_data['sampled_gazes'][idx].append(current_gaze)
                        sample_count += 1
                else:
                    cv2.imshow(self.gaze_window, gaze_canvas)
                    cv2.waitKey(10)

            # 确保采样10次，如果不足则填充None
            while len(self.experiment_data['sampled_gazes'][idx]) < 10:
                self.experiment_data['sampled_gazes'][idx].append(None)

            # 计算平均误差
            errors = []
            for gaze in self.experiment_data['sampled_gazes'][idx]:
                if gaze:
                    error = np.sqrt((gaze[0] - target_x)**2 + (gaze[1] - target_y)**2)
                    errors.append(error)
            if errors:
                avg_error = np.mean(errors)
                self.experiment_data['errors'].append(avg_error)
                print(f"点 {idx+1} 的平均误差: {avg_error:.1f} 像素")
            else:
                print(f"点 {idx+1} 没有有效采样")
                self.experiment_data['errors'].append(None)

        # 实验结束后绘制结果
        self.plot_experiment_results()

    def plot_experiment_results(self):
        """绘制所有目标点和采样点，并保存图片"""
        canvas = self.create_fullscreen_canvas()
        for idx, (target_x, target_y) in enumerate(self.target_points):
            color = self.colors[idx]
            # 绘制目标点
            cv2.circle(canvas, (target_x, target_y), self.target_radius, color, -1)
            # 绘制采样点
            for gaze in self.experiment_data['sampled_gazes'][idx]:
                if gaze:
                    cv2.circle(canvas, gaze, self.gaze_radius, color, 2)
        # 保存图片
        cv2.imwrite("accuracy_experiment.png", canvas)
        print("实验结果已保存为 accuracy_experiment.png")

    def run(self):
        """修改后的主运行循环"""
        webcam = cv2.VideoCapture(0)
        self.init_windows()
        cv2.imshow(self.gaze_window, self.create_fullscreen_canvas())
        cv2.waitKey(100)

        # 校准阶段
        self.calibrate(webcam)

        # 实验阶段
        if self.calibration_data['calibrated']:
            self.run_accuracy_experiment(webcam)

        # 实时注视点预测（保持原有逻辑）
        while True:
            _, frame = webcam.read()
            frame = self.apply_clahe(frame)
            current_gaze = None

            # 获取特征
            avg_v = self.get_eye_features(frame)
            if avg_v:
                current_gaze = self.predict_gaze(avg_v[0], avg_v[1])
                self.consecutive_failures = 0
            else:
                self.consecutive_failures += 1

            # 显示逻辑（保持原样）
            if self.consecutive_failures > 15:
                gaze_canvas = self.create_fullscreen_canvas()
                cv2.putText(gaze_canvas, "Tracking Lost! Please face the camera",
                           (self.screen_w//4, self.screen_h//2), cv2.FONT_HERSHEY_SIMPLEX,
                           1.5, (255, 255, 255), 3)
            else:
                gaze_canvas = self.create_fullscreen_canvas()
                for x, y in self.target_points:
                    cv2.circle(gaze_canvas, (x, y), self.target_radius,
                               self.target_color, -1)
                    cv2.circle(gaze_canvas, (x, y), self.target_radius + 5,
                               (255, 255, 255), 2)
                if current_gaze:
                    cv2.circle(gaze_canvas, current_gaze, self.gaze_radius,
                               self.gaze_color, -1)
                    cv2.putText(gaze_canvas,
                                f"({current_gaze[0]}, {current_gaze[1]})",
                                (current_gaze[0] + 50, current_gaze[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.setWindowProperty(self.gaze_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow(self.gaze_window, gaze_canvas)
            cv2.imshow(self.camera_window, frame)

            if cv2.waitKey(1) == 27:
                break

        webcam.release()
        cv2.destroyAllWindows()

    # 以下保持原有方法完全不变
    def init_kalman_filter(self):
        """初始化卡尔曼滤波器"""
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-4
        kalman.statePost = np.array([self.screen_w//2, self.screen_h//2, 0, 0], dtype=np.float32)
        return kalman

    def get_screen_resolution(self):
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return {'width': width, 'height': height}

    def init_windows(self):
        cv2.namedWindow(self.camera_window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.camera_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.moveWindow(self.camera_window, 0, 0)
        cv2.namedWindow(self.gaze_window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.gaze_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(self.gaze_window, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_FREERATIO)
        cv2.namedWindow(self.calibration_window, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.calibration_window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

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
                int(np.clip(predicted[1], 0, self.screen_h)))
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

if __name__ == "__main__":
    visualizer = GazeVisualizer()
    visualizer.run()