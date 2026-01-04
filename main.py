import cv2
import time
import os
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from fps_monitor import FpsMonitor
from object_tracker import ObjectTracker
from ultralytics import YOLO
from analyze import analyze_tracked_object, get_crossing_frame
from ollama import read_plate
from mqtt_integration import get_mqtt_publisher
from ffmpeg_capture import FFmpegCapture

load_dotenv()

# Configuration
HEADLESS = True  # Set to False to show GUI window

# Initialize publisher (choose one)
mqtt = get_mqtt_publisher()  # MQTT version

# Frame dimensions (set after first frame)
frame_width = None
frame_height = None

# Load known plates for recognition guidance
KNOWN_PLATES_STR = os.getenv("KNOWN_PLATES", "")
KNOWN_PLATES = [p.strip().upper() for p in KNOWN_PLATES_STR.split(",") if p.strip()] if KNOWN_PLATES_STR else None
if KNOWN_PLATES:
    print(f"Known plates loaded: {', '.join(KNOWN_PLATES)}")

print(cv2.getBuildInformation())

def on_car_lost(obj):
    global frame_width, frame_height

    duration = obj.duration()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if duration < 1.0:
        print(f"[{now}] Object #{obj.track_id} ignored (duration {duration:.2f}s < 1s)")
        return

    frame_count = len(obj.boxes)

    # Get HD frames for this object
    hd_frames = tracker.get_frames_for_object(obj)
    hd_frame_count = len(hd_frames)

    # Debug: Create video from all HD frames
    if hd_frame_count > 0:
        video_path = f"./frames/video_{obj.track_id}.mp4"
        first_frame = hd_frames[0]
        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 10.0, (w, h))

        for frame in hd_frames:
            video_writer.write(frame)

        video_writer.release()
        print(f"  Debug - Saved video with {hd_frame_count} frames to {video_path}")

    analysis = analyze_tracked_object(obj, frame_width, frame_height)

    print(f"[{now}] Car #{obj.track_id}: {analysis['action']} - "
            f"tracked for {duration:.1f}s ({frame_count} frames, {hd_frame_count} HD)")

    if analysis['crossed_at']:
        crossing_time = analysis['crossed_at'] - obj.first_seen
        print(f"  Crossed 75% line at {crossing_time:.2f}s "
                f"(from {'right' if analysis['crossed_from_right'] else 'left'})")

        # Get the frame index where crossing occurred
        crossing_frame_idx = get_crossing_frame(obj, frame_width)

        if crossing_frame_idx is not None and crossing_frame_idx < len(obj.frame_ids):
            # Get the frame ID at the crossing point
            crossing_frame_id = obj.frame_ids[crossing_frame_idx]

            # Get the HD frame at the crossing point (no sync issues - same source!)
            if crossing_frame_id in tracker.hd_frames:
                hd_frame = tracker.hd_frames[crossing_frame_id]
                box = obj.boxes[crossing_frame_idx]

                hd_height, hd_width = hd_frame.shape[:2]

                # Calculate scale factors from low-res tracking to HD
                scale_x = hd_width / TARGET_WIDTH
                scale_y = hd_height / TARGET_HEIGHT

                # Extract bounding box coordinates (xywh format) from low-res tracking
                x_center, y_center, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])

                # Scale to HD frame coordinates
                x_center_hd = x_center * scale_x
                y_center_hd = y_center * scale_y
                w_hd = w * scale_x
                h_hd = h * scale_y

                # Convert from center+size to corner coordinates
                x1 = int(x_center_hd - w_hd / 2)
                y1 = int(y_center_hd - h_hd / 2)
                x2 = int(x_center_hd + w_hd / 2)
                y2 = int(y_center_hd + h_hd / 2)

                # Ensure coordinates are within HD frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(hd_width, x2)
                y2 = min(hd_height, y2)

                # Debug: Print coordinates
                print(f"  Debug - Low-res box (xywh): center=({x_center:.1f},{y_center:.1f}) size=({w:.1f},{h:.1f})")
                print(f"  Debug - HD frame size: {hd_width}x{hd_height}, Tracking: {TARGET_WIDTH}x{TARGET_HEIGHT}")
                print(f"  Debug - Scale factors: x={scale_x:.2f}, y={scale_y:.2f}")
                print(f"  Debug - HD box corners: ({x1},{y1}) to ({x2},{y2})")
                print(f"  Debug - Crop size: {x2-x1}x{y2-y1}")

                # Debug: Save full HD frame with bounding box drawn
                debug_frame = hd_frame.copy()

                # Draw 75% vertical line (scaled to HD)
                vertical_line_x = int(hd_width * 0.75)
                cv2.line(debug_frame, (vertical_line_x, 0), (vertical_line_x, hd_height),
                        (0, 255, 255), 4)

                # Draw bounding box
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

                # Add labels
                cv2.putText(debug_frame, f"Track {obj.track_id} - Frame {crossing_frame_id}", (x1, y1-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                cv2.putText(debug_frame, f"Crop: {x2-x1}x{y2-y1}px", (x1, y2+40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

                cv2.imwrite(f"./frames/debug_full_{obj.track_id}.jpg", debug_frame)

                cropped_car = hd_frame[y1:y2, x1:x2]
                cv2.imwrite(f"./frames/cropped_{obj.track_id}.jpg", cropped_car)

                # Read the license plate
                print(f"  Reading license plate from HD frame {crossing_frame_id}...")
                plate_number = read_plate(cropped_car, known_plates=KNOWN_PLATES)

                if plate_number:
                    print(f"  License Plate: {plate_number}")
                else:
                    print(f"  License Plate: Could not read")
                    plate_number = None
            else:
                print(f"  No HD frame available at crossing point")
                plate_number = None
        else:
            plate_number = None
    else:
        plate_number = None

    # Publish to MQTT
    detection_time = datetime.fromtimestamp(obj.first_seen)
    mqtt.publish_detection(plate_number, analysis['action'], detection_time)

    print(f"  Movement: dx={analysis['direction']['delta_x']:.1f}, "
            f"dy={analysis['direction']['delta_y']:.1f}")
    print(f"  Frame IDs: {obj.frame_ids[:5]}... (showing first 5)")

model = YOLO("yolo11n.pt")

tracker = ObjectTracker(
    class_id=2, #Cars only
    frames_before_purge=20,
    on_object_lost=on_car_lost
)

stream_url = os.getenv("CAM_DRIVEWAY_HIGH")

# Target dimensions for downsampled frames (for YOLO tracking)
TARGET_WIDTH = 640
TARGET_HEIGHT = 360

def open_stream(url, name):
    """Open a video stream with error handling using FFmpeg subprocess"""
    print(f"Opening {name} stream with FFmpeg...")
    cap = FFmpegCapture(url)

    # Wait for first frame (with timeout)
    max_wait = 30  # Wait up to 30 seconds for first frame
    wait_interval = 0.5
    waited = 0

    while waited < max_wait:
        if cap.grab():
            print(f"{name} stream opened successfully")
            return cap
        time.sleep(wait_interval)
        waited += wait_interval

    print(f"Error: Could not open {name} stream - timeout waiting for first frame")
    exit(1)

def reconnect_stream(cap, url, name):
    """Reconnect to a stream that has failed"""
    print(f"Reconnecting to {name} stream...")
    cap.release()
    time.sleep(1)  # Brief pause before reconnecting
    cap = FFmpegCapture(url)
    time.sleep(2)  # Give FFmpeg time to start
    if cap.isOpened():
        print(f"{name} stream reconnected successfully")
    else:
        print(f"Failed to reconnect to {name} stream")
    return cap

# Open HD stream - we'll downsample in memory for YOLO tracking
cap = open_stream(stream_url, "HD camera")

print("Stream opened successfully!")

fps = FpsMonitor()
if HEADLESS:
    print("Starting stream in headless mode... Press Ctrl+C to quit")
else:
    print("Starting stream... Press 'q' to quit")

consecutive_errors = 0
max_errors_before_reconnect = 10  # Try reconnecting after this many errors
max_reconnect_attempts = 3  # Maximum reconnection attempts before giving up
reconnect_count = 0

while True:
    # Grab and retrieve frame from HD stream
    try:
        if not cap.grab():
            consecutive_errors += 1
            if consecutive_errors >= max_errors_before_reconnect:
                if reconnect_count < max_reconnect_attempts:
                    cap = reconnect_stream(cap, stream_url, "HD camera")
                    consecutive_errors = 0
                    reconnect_count += 1
                    time.sleep(2)
                    continue
                else:
                    print(f"Max reconnection attempts ({max_reconnect_attempts}) exceeded, exiting")
                    break
            time.sleep(0.1)
            continue

        success, frame_hd = cap.retrieve()
        if not success or frame_hd is None:
            consecutive_errors += 1
            if consecutive_errors >= max_errors_before_reconnect:
                if reconnect_count < max_reconnect_attempts:
                    cap = reconnect_stream(cap, stream_url, "HD camera")
                    consecutive_errors = 0
                    reconnect_count += 1
                    time.sleep(2)
                    continue
                else:
                    print(f"Max reconnection attempts ({max_reconnect_attempts}) exceeded, exiting")
                    break
            time.sleep(0.1)
            continue

    except Exception as e:
        print(f"Error grabbing frame: {e}")
        consecutive_errors += 1
        if consecutive_errors >= max_errors_before_reconnect:
            if reconnect_count < max_reconnect_attempts:
                cap = reconnect_stream(cap, stream_url, "HD camera")
                consecutive_errors = 0
                reconnect_count += 1
                time.sleep(2)
                continue
            else:
                print(f"Max reconnection attempts ({max_reconnect_attempts}) exceeded, exiting")
                break
        time.sleep(0.1)
        continue

    # Reset error counters on success
    consecutive_errors = 0
    reconnect_count = 0

    # Set frame dimensions on first frame (based on target low-res size)
    if frame_width is None:
        frame_width = TARGET_WIDTH
        frame_height = TARGET_HEIGHT
        hd_height, hd_width = frame_hd.shape[:2]
        print(f"HD frame dimensions: {hd_width}x{hd_height}")
        print(f"Tracking frame dimensions: {frame_width}x{frame_height}")

    # Downsample HD frame to low-res for YOLO tracking
    frame_low = cv2.resize(frame_hd, (TARGET_WIDTH, TARGET_HEIGHT))

    # Apply triangular mask to top right corner (to ignore timestamp/overlay)
    # Triangle extends 10% left from right edge and 40% down from top
    mask_points = np.array([
        [frame_width, 0],                           # Top right corner
        [int(frame_width * 0.1), 0],                # 10% left along top edge
        [frame_width, int(frame_height * 0.40)]     # 40% down along right edge
    ], dtype=np.int32)
    cv2.fillPoly(frame_low, [mask_points], (0, 0, 0))

    # Run YOLO tracking on downsampled frame
    results = model.track(frame_low, persist=True, verbose=False)
    tracked_objects = tracker.update(results[0], hd_frame=frame_hd)

    # Update FPS counter
    fps.tick()

    if not HEADLESS:
        annotated_frame = results[0].plot()

        # Draw 75% vertical line
        vertical_line_x = int(frame_width * 0.75)
        cv2.line(annotated_frame, (vertical_line_x, 0), (vertical_line_x, frame_height),
                 (0, 255, 255), 2)

        # FPS Display
        cv2.putText(annotated_frame, f"FPS: {fps.get_fps():.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display tracking count
        cv2.putText(annotated_frame, f"Tracking: {len(tracked_objects)} cars", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Driveway Camera Stream", annotated_frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Small delay to prevent CPU spinning in headless mode
        time.sleep(0.01)

# Release resources
cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()
mqtt.disconnect()
print("Stream ended")
