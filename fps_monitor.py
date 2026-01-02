import time
from collections import deque


class FpsMonitor:
    """Simple FPS monitor that calculates average FPS over a sliding window of frames."""

    def __init__(self, window_size=90):
        """
        Initialize the FPS monitor.

        Args:
            window_size: Number of frames to use for FPS calculation (default: 10)
        """
        self.window_size = window_size
        self.timestamps = deque(maxlen=window_size)

    def tick(self):
        """Record a frame timestamp."""
        self.timestamps.append(time.time())

    def get_fps(self):
        """
        Calculate and return the current FPS.

        Returns:
            float: Average FPS over the last window_size frames, or 0.0 if insufficient data
        """
        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed == 0:
            return 0.0

        return (len(self.timestamps) - 1) / elapsed

    def reset(self):
        """Clear all timestamps."""
        self.timestamps.clear()
