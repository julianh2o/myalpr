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

    def __init__(self, url, width=None, height=None, max_retries=None, retry_delay=2.0):
        self.url = url
        self.width = width
        self.height = height
        self.process = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.max_retries = max_retries  # None = infinite retries
        self.retry_delay = retry_delay
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

    def _restart_stream(self):
        """Restart the ffmpeg process (called from background thread)"""
        print("Restarting FFmpeg stream...")

        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-rtsp_transport', 'tcp',
            '-i', self.url,
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',
        ]

        if self.width and self.height:
            cmd.extend(['-s', f'{self.width}x{self.height}'])

        cmd.append('pipe:1')

        # Start new ffmpeg process
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=10**8
        )

        print("FFmpeg stream restarted")

    def _read_frames(self):
        """Background thread to read frames from ffmpeg"""
        retry_count = 0
        current_retry_delay = self.retry_delay

        while self.running:
            try:
                raw_frame = self.process.stdout.read(self.frame_size)
                if len(raw_frame) != self.frame_size:
                    print(f"FFmpeg stream ended or error occurred")

                    # Check if we should retry
                    if self.max_retries is not None and retry_count >= self.max_retries:
                        print(f"Max retries ({self.max_retries}) reached, stopping stream")
                        break

                    # Clean up old process
                    if self.process:
                        try:
                            self.process.terminate()
                            self.process.wait(timeout=2.0)
                        except:
                            self.process.kill()

                    # Wait before retrying with exponential backoff
                    retry_count += 1
                    print(f"Retrying in {current_retry_delay}s (attempt {retry_count})...")
                    time.sleep(current_retry_delay)
                    current_retry_delay = min(current_retry_delay * 1.5, 30.0)  # Cap at 30s

                    # Restart the stream
                    if self.running:
                        self._restart_stream()
                        continue
                    else:
                        break

                # Successfully read frame - reset retry counter
                retry_count = 0
                current_retry_delay = self.retry_delay

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

                # Clean up and retry
                if self.max_retries is not None and retry_count >= self.max_retries:
                    print(f"Max retries ({self.max_retries}) reached, stopping stream")
                    break

                if self.process:
                    try:
                        self.process.terminate()
                        self.process.wait(timeout=2.0)
                    except:
                        self.process.kill()

                retry_count += 1
                print(f"Retrying in {current_retry_delay}s (attempt {retry_count})...")
                time.sleep(current_retry_delay)
                current_retry_delay = min(current_retry_delay * 1.5, 30.0)

                if self.running:
                    self._restart_stream()
                else:
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
