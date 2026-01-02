import cv2
import time
import os
from datetime import datetime
from dotenv import load_dotenv
from fps_monitor import FpsMonitor
from object_tracker import ObjectTracker
from ultralytics import YOLO
from analyze import analyze_tracked_object, get_crossing_frame
from ollama import read_plate
from mqtt_integration import get_mqtt_publisher

load_dotenv()

# Configuration
HEADLESS = True  # Set to False to show GUI window

# Initialize publisher (choose one)
mqtt = get_mqtt_publisher()  # MQTT version

# Frame dimensions (set after first frame)
frame_width = None
frame_height = None

def on_car_lost(obj):
    global frame_width, frame_height

    duration = obj.duration()
    frame_count = len(obj.boxes)

    # Get HD frames for this object
    hd_frames = tracker.get_frames_for_object(obj)
    hd_frame_count = len(hd_frames)

    analysis = analyze_tracked_object(obj, frame_width, frame_height)

    print(f"Car #{obj.track_id}: {analysis['action']} - "
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

            # Get the HD frame at the crossing point
            if crossing_frame_id in tracker.hd_frames:
                hd_frame = tracker.hd_frames[crossing_frame_id]
                box = obj.boxes[crossing_frame_idx]

                hd_height, hd_width = hd_frame.shape[:2]
                scale_x, scale_y = hd_width / frame_width, hd_height / frame_height

                # Extract bounding box coordinates (xywh format) from low quality tracking
                x, y, w, h = float(box[0]*scale_x), float(box[1]*scale_y), float(box[2]*scale_x), float(box[3]*scale_y)

                # Convert to pixel coordinates
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)

                # Ensure coordinates are within HD frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(hd_width, x2)
                y2 = min(hd_height, y2)

                cropped_car = hd_frame[y1:y2, x1:x2]
                cv2.imwrite("./cropped.jpg", cropped_car)

                # Read the license plate
                print(f"  Reading license plate from HD frame {crossing_frame_id}...")
                plate_number = read_plate(cropped_car)

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

stream_high = os.getenv("CAM_DRIVEWAY_HIGH")
stream_low = os.getenv("CAM_DRIVEWAY_LOW")

# Set environment variable to skip TLS verification for RTSP streams
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|tls_verify;0'

# Open both low and high quality streams with options
caplow = cv2.VideoCapture(stream_low, cv2.CAP_FFMPEG)
caphigh = cv2.VideoCapture(stream_high, cv2.CAP_FFMPEG)

if not caplow.isOpened():
    print("Error: Could not open low quality stream")
    exit(1)

if not caphigh.isOpened():
    print("Error: Could not open high quality stream")
    exit(1)

print("Both streams opened successfully!")

fps = FpsMonitor()
if HEADLESS:
    print("Starting stream in headless mode... Press Ctrl+C to quit")
else:
    print("Starting stream... Press 'q' to quit")

consecutive_errors = 0
max_errors = 30  # Allow 30 consecutive errors before giving up

while True:
    # Grab from both streams
    try:
        grabbed_low = caplow.grab()
        grabbed_high = caphigh.grab()
    except Exception as e:
        print(f"Error grabbing frames: {e}")
        consecutive_errors += 1
        if consecutive_errors >= max_errors:
            print("Too many consecutive errors, exiting")
            break
        time.sleep(0.1)
        continue

    # Retrieve from low quality stream first
    success_low, frame_low = caplow.retrieve()

    if not success_low or frame_low is None:
        consecutive_errors += 1
        if consecutive_errors >= max_errors:
            print("Too many consecutive frame failures, exiting")
            break
        time.sleep(0.1)
        continue

    # Reset error counter on success
    consecutive_errors = 0

    # Set frame dimensions on first frame
    if frame_width is None:
        frame_height, frame_width = frame_low.shape[:2]
        print(f"Frame dimensions: {frame_width}x{frame_height}")

    # Run YOLO tracking on low quality stream
    results = model.track(frame_low, persist=True, verbose=False)
    tracked_objects = tracker.update(results[0])

    # If there are tracked objects, retrieve and store HD frame
    if len(tracked_objects) > 0 and grabbed_high:
        success_high, frame_high = caphigh.retrieve()
        if not success_high: raise
        tracker.assignFrame(frame_high)

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
caplow.release()
caphigh.release()
if not HEADLESS:
    cv2.destroyAllWindows()
mqtt.disconnect()
print("Stream ended")
