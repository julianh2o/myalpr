import numpy as np


def analyze_tracked_object(tracked_obj, frame_width, frame_height, vertical_line_percent=0.75):
    """
    Analyze a tracked object to determine action (arriving/departing) and crossing point.

    Args:
        tracked_obj: TrackedObject instance with box history
        frame_width: Width of the video frame
        frame_height: Height of the video frame
        vertical_line_percent: Percentage of width for the vertical crossing line (default: 0.75)

    Returns:
        dict: {
            'action': 'arriving' or 'departing',
            'crossed_at': timestamp when left edge crossed the vertical line (or None),
            'direction': dict with movement stats
        }
    """
    if len(tracked_obj.boxes) < 2:
        return {
            'action': 'unknown',
            'crossed_at': None,
            'direction': None
        }

    # Calculate vertical line position
    vertical_line = frame_width * vertical_line_percent

    # Get first and last positions
    first_box = tracked_obj.boxes[0]
    last_box = tracked_obj.boxes[-1]

    # Extract center positions (boxes are in xywh format)
    first_x, first_y = float(first_box[0]), float(first_box[1])
    last_x, last_y = float(last_box[0]), float(last_box[1])

    # Calculate movement direction
    delta_x = last_x - first_x
    delta_y = last_y - first_y

    # Determine action based on general movement trend
    # Arriving: moves from top-right to bottom-left (negative x, positive y)
    # Departing: moves from bottom-left to top-right (positive x, negative y)
    if delta_x < 0 and delta_y > 0:
        action = 'arriving'
    elif delta_x > 0 and delta_y < 0:
        action = 'departing'
    else:
        # For ambiguous cases, use the dominant direction
        if abs(delta_x) > abs(delta_y):
            action = 'departing' if delta_x > 0 else 'arriving'
        else:
            action = 'arriving' if delta_y > 0 else 'departing'

    # Find when the left extent crossed the vertical line
    crossed_at = None
    crossed_from_right = None

    for i, box in enumerate(tracked_obj.boxes):
        x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        left_edge = x - w / 2

        if i == 0:
            # Determine which side we started on
            crossed_from_right = left_edge > vertical_line
            continue

        prev_box = tracked_obj.boxes[i - 1]
        prev_x, prev_w = float(prev_box[0]), float(prev_box[2])
        prev_left_edge = prev_x - prev_w / 2

        # Check if we crossed the line
        if crossed_from_right and prev_left_edge > vertical_line and left_edge <= vertical_line:
            # Crossed from right to left
            # Calculate approximate timestamp based on frame timing
            time_per_frame = tracked_obj.duration() / len(tracked_obj.boxes)
            crossed_at = tracked_obj.first_seen + (i * time_per_frame)
            break
        elif not crossed_from_right and prev_left_edge < vertical_line and left_edge >= vertical_line:
            # Crossed from left to right
            time_per_frame = tracked_obj.duration() / len(tracked_obj.boxes)
            crossed_at = tracked_obj.first_seen + (i * time_per_frame)
            break

    return {
        'action': action,
        'crossed_at': crossed_at,
        'direction': {
            'delta_x': delta_x,
            'delta_y': delta_y,
            'start_pos': (first_x, first_y),
            'end_pos': (last_x, last_y)
        },
        'crossed_from_right': crossed_from_right
    }


def get_crossing_frame(tracked_obj, frame_width, vertical_line_percent=0.75):
    """
    Get the frame where the left edge of the bounding box crossed the vertical line.

    Args:
        tracked_obj: TrackedObject instance with box history
        frame_width: Width of the video frame
        vertical_line_percent: Percentage of width for the vertical line (default: 0.75)

    Returns:
        int: Frame index where crossing occurred, or None if no crossing detected
    """
    vertical_line = frame_width * vertical_line_percent

    if len(tracked_obj.boxes) < 2:
        return None

    # Determine initial side
    first_box = tracked_obj.boxes[0]
    first_x, first_w = float(first_box[0]), float(first_box[2])
    first_left_edge = first_x - first_w / 2
    crossed_from_right = first_left_edge > vertical_line

    for i in range(1, len(tracked_obj.boxes)):
        box = tracked_obj.boxes[i]
        x, w = float(box[0]), float(box[2])
        left_edge = x - w / 2

        prev_box = tracked_obj.boxes[i - 1]
        prev_x, prev_w = float(prev_box[0]), float(prev_box[2])
        prev_left_edge = prev_x - prev_w / 2

        # Check for crossing
        if crossed_from_right and prev_left_edge > vertical_line and left_edge <= vertical_line:
            return i
        elif not crossed_from_right and prev_left_edge < vertical_line and left_edge >= vertical_line:
            return i

    return None
