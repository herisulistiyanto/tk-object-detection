import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

try:
    import ttkbootstrap as tb
    BOOTSTRAP = True
except ImportError:
    BOOTSTRAP = False

from ultralytics import YOLO

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Available models: https://docs.ultralytics.com/models/
MODEL_NAME = 'yolo12n.pt'
model = YOLO(MODEL_NAME)

def list_cameras(max_idx=5):
    cameras = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()
    return cameras if cameras else [0]

class StatusBar(ttk.Label):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.set("Ready")
    def set(self, message):
        self.config(text=message)
        self.update_idletasks()

class ObjectDetectorApp:
    def __init__(self, window, title):
        self.window = window
        self.window.title(title)
        self.running = False
        self.vid = None
        self.detect_thread = None
        self.score_threshold = 0.5

        # Main Frame
        self.frame = ttk.Frame(window)
        self.frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Canvas for video
        self.canvas = tk.Canvas(
            self.frame, width=FRAME_WIDTH, height=FRAME_HEIGHT, bd=1, relief="ridge"
        )
        self.canvas.grid(row=0, column=0, columnspan=4, pady=(0, 10))

        # Camera selection
        tk.Label(self.frame, text="Camera:").grid(row=1, column=0, sticky='e')
        cam_idx = list_cameras()
        self.cam_var = tk.IntVar(value=cam_idx[0])
        self.combo_cam = ttk.Combobox(
            self.frame, values=cam_idx, textvariable=self.cam_var, state="readonly", width=6
        )
        self.combo_cam.grid(row=1, column=1, sticky='w')

        # Score threshold slider
        self.slider = ttk.Scale(
            self.frame,
            from_=0.1,
            to=1.0,
            orient=tk.HORIZONTAL,
            value=self.score_threshold,
            command=self.on_threshold_change,
            length=160
        )
        self.slider.grid(row=1, column=2, sticky='w', padx=(10, 0))
        self.slider_label = ttk.Label(self.frame, text=f"Threshold: {self.score_threshold:.2f}")
        self.slider_label.grid(row=1, column=3, sticky='w')

        # Start/Stop button
        self.btn_start = ttk.Button(
            self.frame, text="Start", width=12, command=self.start
        )
        self.btn_start.grid(row=2, column=0, pady=5)
        self.btn_stop = ttk.Button(
            self.frame, text="Stop", width=12, command=self.stop, state=tk.DISABLED
        )
        self.btn_stop.grid(row=2, column=1, pady=5)

        # FPS + Model label
        self.lbl_fps = ttk.Label(self.frame, text="FPS: 0.00")
        self.lbl_fps.grid(row=2, column=2, sticky='w')
        self.lbl_model = ttk.Label(self.frame, text=f"Model: {MODEL_NAME}")
        self.lbl_model.grid(row=2, column=3, sticky='w')

        # Status bar
        self.status = StatusBar(self.frame, anchor="w", relief="sunken")
        self.status.grid(row=3, column=0, columnspan=4, sticky='we', pady=(5,0))

        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def log(self, message):
        self.status.set(message)

    def on_threshold_change(self, value):
        self.score_threshold = float(value)
        self.slider_label.config(text=f"Threshold: {self.score_threshold:.2f}")

    def start(self):
        if self.running:
            return
        self.running = True
        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        cam_idx = self.cam_var.get()
        try:
            self.vid = cv2.VideoCapture(cam_idx)
            self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            if not self.vid.isOpened():
                raise ValueError(f"Camera {cam_idx} not accessible.")
            self.log(f"Camera {cam_idx} started")
            self.detect_thread = threading.Thread(target=self.run_detection, daemon=True)
            self.detect_thread.start()
        except Exception as e:
            self.log(str(e))
            self.running = False
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            if self.vid:
                self.vid.release()
                self.vid = None

    def stop(self):
        self.running = False
        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        if self.vid:
            self.vid.release()
            self.vid = None

    def run_detection(self):
        prev_time = time.time()
        while self.running and self.vid and self.vid.isOpened():
            ret, frame = self.vid.read()
            if not ret:
                self.log("Failed to read from Webcam")
                break
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            try:
                # YOLOv8 predict (return is List[Results])
                results = model.predict(frame, conf=self.score_threshold, verbose=False)
                annotated_frame = results[0].plot()
            except Exception as e:
                self.log(f"Detection error: {str(e)}")
                break

            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            self.lbl_fps.config(text=f"FPS: {fps:.2f}")

            # Show Image on Tkinter Canvas
            img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img_tk = ImageTk.PhotoImage(image=img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.window.update_idletasks()
            self.window.update()
            self.canvas.imgtk = img_tk

        self.stop()
        self.log("Stopped detection thread")

    def on_closing(self):
        self.stop()
        self.log("App closed by user")
        self.window.destroy()

if __name__ == '__main__':
    if BOOTSTRAP:
        root = tb.Window(themename="superhero")
    else:
        root = tk.Tk()
    app = ObjectDetectorApp(root, "YOLOv8 Object Detection App")
    root.mainloop()