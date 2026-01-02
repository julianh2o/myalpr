"""
FFmpeg-based video capture that bypasses OpenCV's RTSP/TLS issues
"""
import subprocess
import numpy as np
import cv2
import threading
import queue


class FFmpegCapture:
    """Video capture using ffmpeg subprocess to avoid OpenCV TLS issues"""

    def __init__(self, url, width=None, height=None):
        self.url = url
        self.width = width
        self.height = height
        self.process = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        self._start()

    def _start(self):
        """Start ffmpeg subprocess and reading thread"""
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', self.url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',  # Disable audio
        ]

        # Add size specification if provided
        if self.width and self.height:
            cmd.extend(['-s', f'{self.width}x{self.height}'])

        cmd.append('pipe:1')

        # Start ffmpeg process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

        # Probe frame size if not provided
        if not self.width or not self.height:
            probe_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=p=0',
                self.url
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                w, h = result.stdout.strip().split(',')
                self.width = int(w)
                self.height = int(h)
            else:
                # Default fallback
                self.width = 640
                self.height = 360

        self.frame_size = self.width * self.height * 3

        # Start reading thread
        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _read_frames(self):
        """Background thread to read frames from ffmpeg"""
        while self.running:
            try:
                raw_frame = self.process.stdout.read(self.frame_size)
                if len(raw_frame) != self.frame_size:
                    print(f"FFmpeg stream ended or error occurred")
                    break

                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))

                # Clear queue if full and add new frame
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                self.frame_queue.put(frame)

            except Exception as e:
                print(f"Error reading frame: {e}")
                break

    def isOpened(self):
        """Check if capture is opened"""
        return self.process is not None and self.process.poll() is None

    def grab(self):
        """Grab frame (compatibility with cv2.VideoCapture)"""
        return not self.frame_queue.empty()

    def retrieve(self):
        """Retrieve grabbed frame"""
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None

    def read(self):
        """Read frame (grab + retrieve)"""
        return self.retrieve()

    def release(self):
        """Release resources"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=2.0)
            if self.process.poll() is None:
                self.process.kill()
