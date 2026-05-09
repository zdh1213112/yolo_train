import sys
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QSpinBox, 
                             QFileDialog, QMessageBox, QGroupBox, QComboBox)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap, QFont

from project_paths import REALSENSE_DATASET_DIR, display_path


class CameraThread(QThread):
    update_frame = pyqtSignal(QImage)
    update_status = pyqtSignal(str)
    update_count = pyqtSignal(int)
    camera_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True
        self.is_recording = False
        self.save_interval = 5
        self.frame_count = 0
        self.frame_counter = 0
        self.base_dir = str(REALSENSE_DATASET_DIR)
        REALSENSE_DATASET_DIR.mkdir(parents=True, exist_ok=True)
        
        # 动态切换控制参数
        self.pipeline = None
        self.align = None
        self.current_serial = None
        self.switch_requested = False

    def request_switch(self, serial):
        """主线程调用：请求切换相机"""
        self.current_serial = serial
        self.switch_requested = True

    def _reinit_camera(self):
        """内部执行：安全重启底层硬件管道"""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None

        if not self.current_serial:
            self.update_status.emit("未选择可用相机")
            return

        self.update_status.emit("正在切换相机，请稍候...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.current_serial)

        # 动态识别设备型号，下发专属分辨率
        ctx = rs.context()
        device_name = "Unknown"
        for d in ctx.query_devices():
            if d.get_info(rs.camera_info.serial_number) == self.current_serial:
                device_name = d.get_info(rs.camera_info.name)
                break

        try:
            # D405 专用微距/全局快门分辨率
            if "D405" in device_name:
                config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
            # D415 / D435 标准分辨率
            else:
                # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

                config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

            self.pipeline.start(config)
            self.align = rs.align(rs.stream.color)
            self.update_status.emit(f"✅ 已连接: {device_name}")
        except Exception as e:
            self.pipeline = None
            self.camera_error.emit(f"相机 [{device_name}] 启动失败: {str(e)}\n请检查占用情况或重新拔插。")
            self.update_status.emit("相机启动失败")

    def run(self):
        while self.running:
            # 监听切换请求
            if self.switch_requested:
                self._reinit_camera()
                self.switch_requested = False

            if not self.pipeline:
                self.msleep(100) # 没有相机时让出CPU
                continue

            try:
                # 阻塞获取帧 (1000ms超时防死锁)
                frames = self.pipeline.wait_for_frames(1000)
            except RuntimeError:
                continue
            except Exception:
                continue

            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image_raw = np.asanyarray(depth_frame.get_data())

            # 数据保存逻辑
            if self.is_recording:
                self.frame_counter += 1
                if self.frame_counter % self.save_interval == 0:
                    self._save_data(color_image, depth_image_raw)

            # UI 画面渲染
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_raw, alpha=0.03), cv2.COLORMAP_JET)
            display_image = np.hstack((color_image, depth_colormap))
            qt_image = self._convert_cv_qt(display_image)
            self.update_frame.emit(qt_image)

        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass

    def _save_data(self, color_image, depth_image_raw):
        color_dir = os.path.join(self.base_dir, "color")
        depth_dir = os.path.join(self.base_dir, "depth")
        os.makedirs(color_dir, exist_ok=True)
        os.makedirs(depth_dir, exist_ok=True)

        filename = f"{self.frame_count:05d}.png"
        cv2.imwrite(os.path.join(color_dir, filename), color_image)
        cv2.imwrite(os.path.join(depth_dir, filename), depth_image_raw)
        
        self.frame_count += 1
        self.update_count.emit(self.frame_count)

    def _convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(960, 360, Qt.KeepAspectRatio)
        return p

    def stop(self):
        self.running = False
        self.wait()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealSense 多设备数据集采集工具")
        self.setFixedSize(1000, 620)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # 先初始化后台线程，后面的默认路径展示要用到它
        self.camera_thread = CameraThread()

        # 1. 设备选择面板
        device_group = QGroupBox("硬件设备选择")
        device_layout = QHBoxLayout()
        self.combo_camera = QComboBox()
        self.combo_camera.setMinimumWidth(300)
        self.btn_refresh = QPushButton("🔄 扫描设备")
        self.btn_refresh.setStyleSheet("font-weight: bold;")
        
        device_layout.addWidget(QLabel("可用相机列表:"))
        device_layout.addWidget(self.combo_camera)
        device_layout.addWidget(self.btn_refresh)
        device_layout.addStretch()
        device_group.setLayout(device_layout)
        self.layout.addWidget(device_group)

        # 2. 视频显示区域
        self.image_label = QLabel("正在初始化系统...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: white; font-size: 16px;")
        self.image_label.setMinimumSize(960, 360)
        self.layout.addWidget(self.image_label)

        # 3. 控制面板
        control_group = QGroupBox("采集控制")
        control_layout = QHBoxLayout()

        self.btn_dir = QPushButton("选择保存目录")
        self.label_dir = QLabel(display_path(self.camera_thread.base_dir))
        self.label_dir.setMinimumWidth(150)

        label_interval = QLabel("保存间隔(帧):")
        self.spin_interval = QSpinBox()
        self.spin_interval.setRange(1, 100)
        self.spin_interval.setValue(5)

        self.btn_start = QPushButton("▶ 开始录制")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_pause = QPushButton("⏸ 暂停录制")
        self.btn_pause.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold;")
        self.btn_pause.setEnabled(False)

        self.label_status = QLabel("状态: 准备就绪")
        self.label_status.setStyleSheet("color: blue; font-weight: bold;")
        self.label_count = QLabel("已保存: 0 组")
        self.label_count.setFont(QFont("Arial", 10, QFont.Bold))

        control_layout.addWidget(self.btn_dir)
        control_layout.addWidget(self.label_dir)
        control_layout.addSpacing(20)
        control_layout.addWidget(label_interval)
        control_layout.addWidget(self.spin_interval)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.btn_start)
        control_layout.addWidget(self.btn_pause)
        control_layout.addStretch()
        control_layout.addWidget(self.label_status)
        control_layout.addSpacing(20)
        control_layout.addWidget(self.label_count)

        control_group.setLayout(control_layout)
        self.layout.addWidget(control_group)

        # 4. 初始化并连接后台线程
        self.camera_thread.update_frame.connect(self.update_image)
        self.camera_thread.update_status.connect(self.update_status)
        self.camera_thread.update_count.connect(self.update_count)
        self.camera_thread.camera_error.connect(self.show_error)
        
        self.btn_start.clicked.connect(self.start_recording)
        self.btn_pause.clicked.connect(self.pause_recording)
        self.spin_interval.valueChanged.connect(self.change_interval)
        self.btn_dir.clicked.connect(self.choose_directory)
        
        self.btn_refresh.clicked.connect(self.refresh_camera_list)
        self.combo_camera.currentIndexChanged.connect(self.on_camera_selected)

        # 启动线程并初次扫描设备
        self.camera_thread.start()
        self.refresh_camera_list()

    def refresh_camera_list(self):
        """扫描当前 USB 端口上所有的 RealSense 设备"""
        self.combo_camera.blockSignals(True)
        self.combo_camera.clear()
        
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            self.combo_camera.addItem("未检测到硬件设备", None)
            self.image_label.setText("请插入 Intel RealSense 相机后点击 [扫描设备]")
        else:
            for dev in devices:
                name = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                self.combo_camera.addItem(f"{name} (SN: {serial})", serial)
        
        self.combo_camera.blockSignals(False)
        
        # 如果检测到设备，自动触发选中第一个
        if self.combo_camera.count() > 0 and self.combo_camera.itemData(0) is not None:
            self.on_camera_selected(0)

    def on_camera_selected(self, index):
        """UI 下拉框切换时，通知底层线程切换设备"""
        serial = self.combo_camera.itemData(index)
        if serial:
            # 切换相机时自动暂停录制，防止数据混乱
            if self.camera_thread.is_recording:
                self.pause_recording()
            self.camera_thread.request_switch(serial)

    @pyqtSlot(QImage)
    def update_image(self, qt_image):
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    @pyqtSlot(str)
    def update_status(self, text):
        self.label_status.setText(f"状态: {text}")

    @pyqtSlot(int)
    def update_count(self, count):
        self.label_count.setText(f"已保存: {count} 组")

    @pyqtSlot(str)
    def show_error(self, error_msg):
        QMessageBox.critical(self, "相机异常", error_msg)
        self.label_status.setText("状态: 设备异常")

    def start_recording(self):
        # 拦截：确保选了相机才允许录制
        if not self.camera_thread.current_serial or not self.camera_thread.pipeline:
            QMessageBox.warning(self, "操作无效", "请先选择并成功连接一台相机！")
            return
            
        self.camera_thread.is_recording = True
        self.camera_thread.save_interval = self.spin_interval.value()
        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.spin_interval.setEnabled(False)
        self.combo_camera.setEnabled(False) # 录制时禁止切换相机
        self.btn_refresh.setEnabled(False)
        self.update_status("正在录制中...")
        self.label_status.setStyleSheet("color: red; font-weight: bold;")

    def pause_recording(self):
        self.camera_thread.is_recording = False
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.spin_interval.setEnabled(True)
        self.combo_camera.setEnabled(True)  # 暂停后允许切换相机
        self.btn_refresh.setEnabled(True)
        self.update_status("已暂停")
        self.label_status.setStyleSheet("color: green; font-weight: bold;")

    def change_interval(self):
        self.camera_thread.save_interval = self.spin_interval.value()

    def choose_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存目录", self.camera_thread.base_dir)
        if dir_path:
            self.camera_thread.base_dir = dir_path
            self.label_dir.setText(display_path(dir_path))

    def closeEvent(self, event):
        self.camera_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
