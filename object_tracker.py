import time
from collections import defaultdict


class TrackedObject:
    """Represents a single tracked object."""

    def __init__(self, track_id, class_id, box):
        self.track_id = track_id
        self.class_id = class_id
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frames_since_seen = 0
        self.boxes = [box]  # History of bounding boxes
        self.frame_ids = []  # Frame IDs where this object appeared
        self.data = {}  # Custom data storage

    def update(self, box, frame_id):
        """Update object with new detection."""
        self.last_seen = time.time()
        self.frames_since_seen = 0
        self.boxes.append(box)
        self.frame_ids.append(frame_id)

    def mark_not_seen(self):
        """Increment frames since last seen."""
        self.frames_since_seen += 1

    def duration(self):
        """Get how long the object has been tracked (in seconds)."""
        return self.last_seen - self.first_seen


class ObjectTracker:
    """Tracks objects across frames with automatic cleanup."""

    def __init__(self, class_id, frames_before_purge=20, on_object_lost=None):
        """
        Initialize the object tracker.

        Args:
            class_id: YOLO class ID to track (e.g., 2 for cars in COCO)
            frames_before_purge: Number of frames before removing unseen objects (default: 20)
            on_object_lost: Callback function called when object is purged, receives TrackedObject
        """
        self.class_id = class_id
        self.frames_before_purge = frames_before_purge
        self.on_object_lost = on_object_lost
        self.tracked_objects = {}  # track_id -> TrackedObject
        self.current_frame = 0
        self.hd_frames = {}  # frame_id -> frame

    def register_callback(self, callback):
        """
        Register a callback to be called when objects are lost.

        Args:
            callback: Function that takes a TrackedObject as parameter
        """
        self.on_object_lost = callback

    def update(self, result, hd_frame=None):
        """
        Update tracker with new YOLO tracking result.

        Args:
            result: YOLO tracking result object
            hd_frame: Optional high-resolution frame to store with this frame ID

        Returns:
            list: List of TrackedObject instances currently being tracked
        """
        self.current_frame += 1
        seen_this_frame = set()

        # Store HD frame if provided
        if hd_frame is not None:
            self.hd_frames[self.current_frame] = hd_frame

        # Process detections
        if result.boxes and result.boxes.is_track:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist()
            clss = result.boxes.cls.int().cpu().tolist()

            # Filter and process objects of the specified class
            for box, track_id, cls in zip(boxes, track_ids, clss):
                if cls == self.class_id:
                    seen_this_frame.add(track_id)

                    if track_id in self.tracked_objects:
                        # Update existing tracked object
                        self.tracked_objects[track_id].update(box, self.current_frame)
                    else:
                        # Create new tracked object
                        obj = TrackedObject(track_id, cls, box)
                        obj.frame_ids.append(self.current_frame)
                        self.tracked_objects[track_id] = obj

        # Mark unseen objects and purge old ones
        to_purge = []
        for track_id, obj in self.tracked_objects.items():
            if track_id not in seen_this_frame:
                obj.mark_not_seen()
                if obj.frames_since_seen >= self.frames_before_purge:
                    to_purge.append(track_id)

        # Purge and call callbacks
        for track_id in to_purge:
            obj = self.tracked_objects.pop(track_id)
            if self.on_object_lost:
                self.on_object_lost(obj)

        # Cleanup unused HD frames
        self._cleanup_hd_frames()

        return list(self.tracked_objects.values())

    def assignFrame(self, frame):
        """
        Assign a high-resolution frame to the current frame ID.
        This should be called after update() to associate the HD frame with the current frame.

        Args:
            frame: High-resolution frame to store
        """
        self.hd_frames[self.current_frame] = frame

    def get_frames_for_object(self, obj):
        """
        Get all HD frames for a tracked object.

        Args:
            obj: TrackedObject instance

        Returns:
            list: List of HD frames corresponding to the object's frame IDs
        """
        return [self.hd_frames[fid] for fid in obj.frame_ids if fid in self.hd_frames]

    def _cleanup_hd_frames(self):
        """Remove HD frames that are no longer referenced by any tracked object."""
        # Collect all frame IDs currently referenced
        referenced_frames = set()
        for obj in self.tracked_objects.values():
            referenced_frames.update(obj.frame_ids)

        # Remove unreferenced frames
        frames_to_remove = [fid for fid in self.hd_frames.keys() if fid not in referenced_frames]
        for fid in frames_to_remove:
            del self.hd_frames[fid]

    def get_tracked_objects(self):
        """Get all currently tracked objects."""
        return list(self.tracked_objects.values())

    def get_object(self, track_id):
        """Get a specific tracked object by ID."""
        return self.tracked_objects.get(track_id)

    def clear(self):
        """Clear all tracked objects."""
        self.tracked_objects.clear()
