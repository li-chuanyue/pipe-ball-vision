import datetime as dt
import os
import queue
import socket
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import imageio_ffmpeg
from PIL import Image, ImageTk


DEFAULT_URL = "rtsp://192.168.137.2:8554/live"
RECORDINGS_DIR = Path.home() / "Videos" / "MaixCAM_Recordings"


class MaixCAMVideoConsole:
    def __init__(self, root):
        self.root = root
        self.root.title("MaixCAM 实时图传与录像")
        self.root.geometry("1180x760")
        self.root.minsize(920, 620)

        self.capture = None
        self.capture_thread = None
        self.capture_stop = threading.Event()
        self.hotspot_stop = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.record_process = None
        self.record_path = None
        self.connected = False
        self.last_frame_time = 0.0
        self.frame_counter = 0
        self.fps_time = time.monotonic()
        self.display_fps = 0.0

        self.url_var = tk.StringVar(value=DEFAULT_URL)
        self.status_var = tk.StringVar(value="等待连接")
        self.metrics_var = tk.StringVar(value="")

        self._build_ui()
        self._refresh_recordings()
        self._update_preview()
        threading.Thread(target=self._hotspot_watchdog, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        toolbar = ttk.Frame(self.root, padding=(10, 10, 10, 8))
        toolbar.pack(fill=tk.X)

        ttk.Label(toolbar, text="RTSP 地址").pack(side=tk.LEFT)
        self.url_entry = ttk.Entry(toolbar, textvariable=self.url_var, width=42)
        self.url_entry.pack(side=tk.LEFT, padx=(8, 8), fill=tk.X, expand=True)

        self.discover_button = ttk.Button(
            toolbar, text="自动搜索", command=self._start_discovery
        )
        self.discover_button.pack(side=tk.LEFT, padx=3)

        self.connect_button = ttk.Button(
            toolbar, text="连接", command=self._toggle_connection
        )
        self.connect_button.pack(side=tk.LEFT, padx=3)

        self.record_button = ttk.Button(
            toolbar, text="开始录像", command=self._toggle_recording, state=tk.DISABLED
        )
        self.record_button.pack(side=tk.LEFT, padx=(10, 3))

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        video_frame = ttk.Frame(body)
        files_frame = ttk.Frame(body, width=330)
        body.add(video_frame, weight=4)
        body.add(files_frame, weight=1)

        self.preview = ttk.Label(
            video_frame,
            text="连接 MaixCAM 后显示实时画面",
            anchor=tk.CENTER,
            background="#111111",
            foreground="#dddddd",
        )
        self.preview.pack(fill=tk.BOTH, expand=True)

        files_toolbar = ttk.Frame(files_frame)
        files_toolbar.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(files_toolbar, text="录像文件").pack(side=tk.LEFT)
        ttk.Button(
            files_toolbar, text="打开目录", command=self._open_recordings_dir
        ).pack(side=tk.RIGHT)

        self.recordings = ttk.Treeview(
            files_frame,
            columns=("time", "size"),
            show="headings",
            selectmode="browse",
        )
        self.recordings.heading("time", text="时间")
        self.recordings.heading("size", text="大小")
        self.recordings.column("time", width=175, anchor=tk.W)
        self.recordings.column("size", width=75, anchor=tk.E)
        self.recordings.pack(fill=tk.BOTH, expand=True)
        self.recordings.bind("<Double-1>", self._play_selected)

        files_actions = ttk.Frame(files_frame)
        files_actions.pack(fill=tk.X, pady=(6, 0))
        ttk.Button(
            files_actions, text="播放", command=self._play_selected
        ).pack(side=tk.LEFT)
        ttk.Button(
            files_actions, text="删除", command=self._delete_selected
        ).pack(side=tk.LEFT, padx=6)
        ttk.Button(
            files_actions, text="刷新", command=self._refresh_recordings
        ).pack(side=tk.RIGHT)

        status = ttk.Frame(self.root, padding=(10, 2, 10, 10))
        status.pack(fill=tk.X)
        ttk.Label(status, textvariable=self.status_var).pack(side=tk.LEFT)
        ttk.Label(status, textvariable=self.metrics_var).pack(side=tk.RIGHT)

    def _set_status(self, text):
        self.root.after(0, self.status_var.set, text)

    def _toggle_connection(self):
        if self.connected or self.capture_thread:
            self._disconnect()
        else:
            self._connect()

    def _connect(self):
        url = self.url_var.get().strip()
        if not url.startswith("rtsp://"):
            messagebox.showerror("地址错误", "请输入 rtsp:// 开头的地址")
            return

        self.capture_stop.clear()
        self.connect_button.configure(text="断开")
        self.discover_button.configure(state=tk.DISABLED)
        self._set_status("正在连接 " + url)
        self.capture_thread = threading.Thread(
            target=self._capture_loop, args=(url,), daemon=True
        )
        self.capture_thread.start()

    def _capture_loop(self, url):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
        )
        capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.capture = capture

        if not capture.isOpened():
            self._set_status("连接失败，请确认 MaixCAM 已推流且网络可达")
            self.root.after(0, self._connection_stopped)
            return

        self.connected = True
        self.last_frame_time = time.monotonic()
        self.root.after(0, lambda: self.record_button.configure(state=tk.NORMAL))
        self._set_status("实时画面已连接")

        while not self.capture_stop.is_set():
            ok, frame = capture.read()
            if not ok:
                if time.monotonic() - self.last_frame_time > 3:
                    self._set_status("视频流中断")
                    break
                time.sleep(0.01)
                continue

            self.last_frame_time = time.monotonic()
            self.frame_counter += 1
            now = time.monotonic()
            elapsed = now - self.fps_time
            if elapsed >= 1.0:
                self.display_fps = self.frame_counter / elapsed
                self.frame_counter = 0
                self.fps_time = now

            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass

        capture.release()
        self.capture = None
        self.connected = False
        self.root.after(0, self._connection_stopped)

    def _connection_stopped(self):
        self.capture_thread = None
        self.connect_button.configure(text="连接")
        self.discover_button.configure(state=tk.NORMAL)
        self.record_button.configure(state=tk.DISABLED)
        if self.record_process:
            self._stop_recording()

    def _disconnect(self):
        self.capture_stop.set()
        self._set_status("正在断开")

    def _update_preview(self):
        try:
            frame = self.frame_queue.get_nowait()
        except queue.Empty:
            frame = None

        if frame is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            width = max(320, self.preview.winfo_width())
            height = max(240, self.preview.winfo_height())
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.preview.configure(image=photo, text="")
            self.preview.image = photo
            self.metrics_var.set(
                "{}×{}  {:.1f} FPS".format(frame.shape[1], frame.shape[0], self.display_fps)
            )

        self.root.after(15, self._update_preview)

    def _toggle_recording(self):
        if self.record_process:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output = RECORDINGS_DIR / f"MaixCAM_{stamp}.mp4"
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "+genpts",
            "-use_wallclock_as_timestamps",
            "1",
            "-i",
            self.url_var.get().strip(),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-an",
            "-movflags",
            "+faststart",
            "-y",
            str(output),
        ]
        try:
            self.record_process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        except OSError as exc:
            messagebox.showerror("录像失败", str(exc))
            self.record_process = None
            return

        self.record_path = output
        self.record_button.configure(text="停止录像")
        self._set_status("正在录像：" + output.name)

    def _stop_recording(self):
        process = self.record_process
        if process is None:
            return

        self.record_process = None
        try:
            if process.stdin:
                process.stdin.write(b"q\n")
                process.stdin.flush()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

        self.record_button.configure(text="开始录像")
        self._set_status("录像已保存：" + str(self.record_path))
        self.record_path = None
        self._refresh_recordings()

    def _refresh_recordings(self):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        for item in self.recordings.get_children():
            self.recordings.delete(item)

        for path in sorted(RECORDINGS_DIR.glob("*.mp4"), reverse=True):
            modified = dt.datetime.fromtimestamp(path.stat().st_mtime)
            size_mb = path.stat().st_size / (1024 * 1024)
            self.recordings.insert(
                "",
                tk.END,
                iid=str(path),
                values=(modified.strftime("%Y-%m-%d %H:%M:%S"), f"{size_mb:.1f} MB"),
            )

    def _selected_path(self):
        selected = self.recordings.selection()
        return Path(selected[0]) if selected else None

    def _play_selected(self, _event=None):
        path = self._selected_path()
        if path and path.exists():
            os.startfile(path)

    def _delete_selected(self):
        path = self._selected_path()
        if not path:
            return
        if messagebox.askyesno("删除录像", "确定删除 {}？".format(path.name)):
            path.unlink(missing_ok=True)
            self._refresh_recordings()

    def _open_recordings_dir(self):
        RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        os.startfile(RECORDINGS_DIR)

    def _start_discovery(self):
        self.discover_button.configure(state=tk.DISABLED)
        self._set_status("正在搜索 MaixCAM RTSP 服务")
        threading.Thread(target=self._discover, daemon=True).start()

    def _discover(self):
        candidates = [f"192.168.137.{index}" for index in range(2, 255)]
        found = None
        for batch_start in range(0, len(candidates), 32):
            if found:
                break
            batch = candidates[batch_start : batch_start + 32]
            result_queue = queue.Queue()

            def probe(host):
                try:
                    with socket.create_connection((host, 8554), timeout=0.18):
                        result_queue.put(host)
                except OSError:
                    pass

            threads = [threading.Thread(target=probe, args=(host,)) for host in batch]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if not result_queue.empty():
                found = result_queue.get()

        if found:
            url = f"rtsp://{found}:8554/live"
            self.root.after(0, self.url_var.set, url)
            self._set_status("发现 MaixCAM：" + found)
        else:
            self._set_status("未发现 MaixCAM，请检查热点、供电和推流程序")
        self.root.after(0, lambda: self.discover_button.configure(state=tk.NORMAL))

    def _hotspot_watchdog(self):
        script = Path(__file__).with_name("start_maixcam_hotspot.ps1")
        while not self.hotspot_stop.is_set():
            subprocess.run(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(script),
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
            self.hotspot_stop.wait(60)

    def _on_close(self):
        self.capture_stop.set()
        self.hotspot_stop.set()
        if self.record_process:
            self._stop_recording()
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        self.root.destroy()


def main():
    root = tk.Tk()
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    MaixCAMVideoConsole(root)
    root.mainloop()


if __name__ == "__main__":
    main()
