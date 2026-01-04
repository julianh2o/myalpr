"""
Rolling frame buffer for video context capture
"""
from collections import deque
import time


class FrameBuffer:
    """Maintains a rolling buffer of recent frames with timestamps"""

    def __init__(self, buffer_seconds=3, estimated_fps=10):
        """
        Initialize frame buffer.

        Args:
            buffer_seconds: How many seconds of frames to keep
            estimated_fps: Estimated frame rate for buffer sizing
        """
        self.buffer_seconds = buffer_seconds
        max_frames = buffer_seconds * estimated_fps * 2  # 2x for safety
        self.buffer = deque(maxlen=max_frames)
        self.current_frame_id = 0

    def add_frame(self, frame):
        """
        Add a frame to the rolling buffer.

        Args:
            frame: The frame to add (numpy array)

        Returns:
            int: The frame ID assigned to this frame
        """
        frame_id = self.current_frame_id
        timestamp = time.time()
        self.buffer.append((frame_id, frame, timestamp))
        self.current_frame_id += 1
        return frame_id

    def get_frames_around(self, start_frame_id, end_frame_id):
        """
        Get frames from buffer before start_frame_id and after end_frame_id.

        Args:
            start_frame_id: First tracked frame ID
            end_frame_id: Last tracked frame ID

        Returns:
            tuple: (frames_before, frames_after) as lists
        """
        frames_before = []
        frames_after = []

        for frame_id, frame, timestamp in self.buffer:
            if frame_id < start_frame_id:
                frames_before.append(frame)
            elif frame_id > end_frame_id:
                frames_after.append(frame)

        return frames_before, frames_after

    def get_current_frame_id(self):
        """Get the current frame ID (next to be assigned)"""
        return self.current_frame_id
