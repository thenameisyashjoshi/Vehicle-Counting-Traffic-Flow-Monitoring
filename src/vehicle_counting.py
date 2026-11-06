import cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
import torch


center_history = defaultdict(lambda:deque(maxlen=5))
previous_centers = {}

# Rolling traffic-flow settings (based on VIDEO time, not processing time).
FLOW_WINDOW_SECONDS = 15
LOW_FLOW_MAX_VPM = 10
MODERATE_FLOW_MAX_VPM = 30

left_crossing_times = deque()
right_crossing_times = deque()
current_video_time = 0.0

# Initial congestion thresholds; calibrate these for the camera/video.
LEFT_DENSITY_MODERATE = 5
LEFT_DENSITY_HIGH = 9
RIGHT_DENSITY_MODERATE = 5
RIGHT_DENSITY_HIGH = 9
MOVEMENT_SLOW_PX_PER_SEC = 12.0
MOVEMENT_STOPPED_PX_PER_SEC = 4.0
MOVEMENT_HISTORY_SECONDS = 1.5
TRACK_ACTIVE_SECONDS = 1.0
track_position_history = defaultdict(lambda: deque())

def get_smooth_center(track_id, new_center):
    center_history[track_id].append(new_center)
    centers = list(center_history[track_id])

    xs = []
    for c in centers:
        x = c[0]
        xs.append(x)

    ys = []
    for c in centers:
        y = c[1]
        ys.append(y)

    total_x = sum(xs)
    total_y = sum(ys)

    count = len(centers)

    avg_x = total_x / count
    avg_y = total_y / count

    smooth_x = int(avg_x)
    smooth_y = int(avg_y)

    smooth_center = (smooth_x, smooth_y)

    return smooth_center

# def is_point_crossing_line(point, line_p1, line_p2, threshold=4):
#     cx,cy = point
#     x1,y1 = line_p1
#     x2,y2 = line_p2

#     A = y2-y1
#     B = x2-x1
#     C = (x2*y1) - (x1*y2)

#     distance = abs(A * cx + B * cy + C) / ((A*A + B*B)**0.5)

#     on_segment = (
#         min(x1,x2) - threshold <= cx <= max(x1,x2) + threshold and min(y1,y2) - threshold <= cy <= max(y1,y2) + threshold)
    
#     return distance <= threshold and on_segment

def has_crossed_line(previous_point, current_point, line_p1, line_p2):
    """
    Returns True when a point moves across the finite line segment.
    """

    if previous_point is None:
        return False

    px, py = previous_point
    cx, cy = current_point

    x1, y1 = line_p1
    x2, y2 = line_p2

    # Signed side of the line for the previous and current positions.
    previous_side = (
        (x2 - x1) * (py - y1)
        - (y2 - y1) * (px - x1)
    )

    current_side = (
        (x2 - x1) * (cy - y1)
        - (y2 - y1) * (cx - x1)
    )

    # No crossing if either point lies exactly on the line.
    # Also, both points must be on opposite sides.
    if previous_side == 0 or current_side == 0:
        return False

    if previous_side * current_side >= 0:
        return False

    # Estimate the intersection between the vehicle's movement
    # and the infinite counting line.
    denominator = previous_side - current_side

    if denominator == 0:
        return False

    t = previous_side / denominator

    intersection_x = px + t * (cx - px)
    intersection_y = py + t * (cy - py)

    # Check that the intersection lies on the finite line segment.
    min_x = min(x1, x2)
    max_x = max(x1, x2)
    min_y = min(y1, y2)
    max_y = max(y1, y2)

    tolerance = 2

    on_segment = (
        min_x - tolerance <= intersection_x <= max_x + tolerance
        and min_y - tolerance <= intersection_y <= max_y + tolerance
    )

    return on_segment




#Bright ROI Overlay
def draw_roi(frame,points):

    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(points,np.int32)], (200, 206, 0))
    return cv2.addWeighted(overlay, 0.25, frame, 0.75,0)

def draw_line(frame, pts):
    cv2.line(frame, pts[0], pts[1], (0, 0, 255), 4)
    return frame


crossed_left_ids = set()
crossed_right_ids = set()


# def draw_bbox(frame, result,left_points, right_points, left_line, right_line, left_count, right_count):
#     global crossed_left_ids, crossed_right_ids

#     for r in result:
#         for box in r.boxes:
#             if box.id is None:
#                 continue

#             track_id = int(box.id)

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             raw_center = ((x1 + x2) // 2, (y1 + y2) // 2)

#             center = get_smooth_center(track_id, raw_center)

#             in_left_region =  cv2.pointPolygonTest(np.array(left_points, np.int32), center , False) >= 0

#             in_right_region = cv2.pointPolygonTest(np.array(right_points, np.int32), center , False) >= 0

#             circle_colour = (0,255,255)

#             if in_left_region:
#                 if track_id not in crossed_left_ids:
#                     if is_point_crossing_line(center, left_line[0], left_line[1]):
#                         left_count +=1
#                         crossed_left_ids.add(track_id)
#                         circle_colour = (255,0,255)
                        
#             elif in_right_region:
#                 if track_id not in crossed_right_ids:
#                     # Check whether the vehicle crossed the right line.
#                     right_count += 1
#                     crossed_right_ids.add(track_id)

#             cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
#             cv2.circle(frame, center, 4, circle_colour, -1)
#             cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)

#     return frame, left_count, right_count
    

def draw_bbox(frame,results,left_points,right_points,left_line,right_line,left_count,right_count,video_time):
    global crossed_left_ids, crossed_right_ids, previous_centers

    for r in results:
        for box in r.boxes:

            # Skip detections without a tracking ID.
            if box.id is None:
                continue

            track_id = int(box.id.item())

            # Get bounding-box coordinates.
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Calculate and smooth the center.
            raw_center = (
                (x1 + x2) // 2,
                (y1 + y2) // 2
            )

            center = get_smooth_center(track_id, raw_center)

            # Store recent smoothed positions for apparent movement estimation.
            motion_history = track_position_history[track_id]
            motion_history.append((video_time, center[0], center[1]))
            while (
                len(motion_history) > 2
                and video_time - motion_history[0][0] > MOVEMENT_HISTORY_SECONDS
            ):
                motion_history.popleft()

            # Get the previous center before updating it.
            previous_center = previous_centers.get(track_id)

            # Check whether the center is inside each ROI.
            in_left_region = (
                cv2.pointPolygonTest(
                    np.array(left_points, np.int32),
                    center,
                    False
                ) >= 0
            )

            in_right_region = (
                cv2.pointPolygonTest(
                    np.array(right_points, np.int32),
                    center,
                    False
                ) >= 0
            )

            circle_colour = (0, 255, 255)

            # LEFT-SIDE COUNTING
            if (
                in_left_region
                and track_id not in crossed_left_ids
            ):
                crossed_left = has_crossed_line(
                    previous_center,
                    center,
                    left_line[0],
                    left_line[1]
                )

                if crossed_left:
                    left_count += 1
                    left_crossing_times.append(current_video_time)
                    crossed_left_ids.add(track_id)
                    circle_colour = (255, 0, 255)

            # RIGHT-SIDE COUNTING
            if (
                in_right_region
                and track_id not in crossed_right_ids
            ):
                crossed_right = has_crossed_line(
                    previous_center,
                    center,
                    right_line[0],
                    right_line[1]
                )

                if crossed_right:
                    right_count += 1
                    right_crossing_times.append(current_video_time)
                    crossed_right_ids.add(track_id)
                    circle_colour = (255, 0, 255)

            # Save this center for the next frame.
            previous_centers[track_id] = center

            # Draw bounding box and center.
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                center,
                4,
                circle_colour,
                -1
            )

            # Display class and tracking ID.
            class_id = int(box.cls.item())
            class_name = r.names[class_id]

            cv2.putText(
                frame,
                f"{class_name} ID:{track_id}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    return frame, left_count, right_count



def update_rolling_flow(crossing_times, video_time, window_seconds=FLOW_WINDOW_SECONDS):
    """Return vehicles/minute over the latest rolling window and its event count."""
    cutoff = video_time - window_seconds

    # Remove crossings that have moved outside the rolling window.
    while crossing_times and crossing_times[0] < cutoff:
        crossing_times.popleft()

    vehicles_in_window = len(crossing_times)
    vehicles_per_minute = vehicles_in_window * 60.0 / window_seconds
    return vehicles_per_minute, vehicles_in_window


def classify_flow(vehicles_per_minute):
    """Simple volume-based label; this is not a direct congestion measurement."""
    if vehicles_per_minute <= LOW_FLOW_MAX_VPM:
        return "LOW FLOW"
    if vehicles_per_minute <= MODERATE_FLOW_MAX_VPM:
        return "MODERATE FLOW"
    return "HIGH FLOW"


def draw_flow_panel(frame, label, vehicles_per_minute, window_count, y):
    """Draw a compact traffic-flow status line on the output frame."""
    text = (
        f"{label}: {vehicles_per_minute:.1f} vehicles/min "
        f"({window_count} in last {FLOW_WINDOW_SECONDS}s)"
    )
    cv2.putText(
        frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.68, (255, 255, 255), 2, cv2.LINE_AA
    )


def point_in_polygon(point, polygon):
    return cv2.pointPolygonTest(
        np.asarray(polygon, dtype=np.int32),
        (float(point[0]), float(point[1])),
        False
    ) >= 0


def estimate_track_movement(track_id):
    history = track_position_history.get(track_id)
    if not history or len(history) < 2:
        return None
    t0, x0, y0 = history[0]
    t1, x1, y1 = history[-1]
    elapsed = t1 - t0
    if elapsed <= 0:
        return None
    return float(np.hypot(x1 - x0, y1 - y0) / elapsed)


def estimate_roi_congestion(roi_polygon, density_moderate, density_high, video_time):
    """Return current ROI vehicle density, apparent movement, and a baseline label."""
    roi_speeds = []
    current_ids = set()

    for track_id, history in track_position_history.items():
        if not history:
            continue
        last_time, cx, cy = history[-1]
        if video_time - last_time > TRACK_ACTIVE_SECONDS:
            continue
        if not point_in_polygon((cx, cy), roi_polygon):
            continue
        current_ids.add(track_id)
        speed = estimate_track_movement(track_id)
        if speed is not None:
            roi_speeds.append(speed)

    density = len(current_ids)
    avg_movement = float(np.mean(roi_speeds)) if roi_speeds else None

    if density >= density_high:
        if avg_movement is not None and avg_movement <= MOVEMENT_STOPPED_PX_PER_SEC:
            label = "SEVERE CONGESTION"
        elif avg_movement is not None and avg_movement <= MOVEMENT_SLOW_PX_PER_SEC:
            label = "HIGH CONGESTION"
        else:
            label = "HIGH DENSITY"
    elif density >= density_moderate:
        if avg_movement is not None and avg_movement <= MOVEMENT_STOPPED_PX_PER_SEC:
            label = "HIGH CONGESTION"
        elif avg_movement is not None and avg_movement <= MOVEMENT_SLOW_PX_PER_SEC:
            label = "MODERATE CONGESTION"
        else:
            label = "MODERATE DENSITY"
    else:
        if density >= 2 and avg_movement is not None and avg_movement <= MOVEMENT_STOPPED_PX_PER_SEC:
            label = "SLOW / QUEUED"
        else:
            label = "LOW CONGESTION"

    return density, avg_movement, label


def draw_congestion_panel(frame, side, density, avg_movement, label, y):
    movement_text = "warming up" if avg_movement is None else f"{avg_movement:.1f} px/s"
    text = f"{side}: {density} vehicles | {label} | Movement: {movement_text}"
    cv2.putText(
        frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.58, (255, 255, 255), 2, cv2.LINE_AA
    )


# Video path
video_path = r"D:/CapstoneProject/Videos/vecteezy_car-and-truck-traffic-on-the-highway-in-europe-poland_7957364.mp4"


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Check your PyTorch installation.")

device = 0

model = YOLO("yolo11x.pt")

print("Using GPU:", torch.cuda.get_device_name(device))


# Open video
cap = cv2.VideoCapture(video_path)

# Check whether video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

print("Success: Video opened successfully!")

# Output video path
output_path = r"D:\CapstoneProject\Videos\vehicle-counting-result.mp4"

# Get input video's FPS before processing
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

# Output dimensions
frame_width = 1280
frame_height = 720

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (frame_width, frame_height)
)

if not out.isOpened():
    cap.release()
    raise RuntimeError("Could not create the output video.")

print("Output video:", output_path)



left_side_points = [(1, 505), (755, 110), (881, 110), (538, 705)]
right_side_points = [ (1277, 323), (886, 641), (932, 113), (1058, 114)]

left_line_points = [(275, 318), (764, 413)]
right_line_points = [(907, 200), (1170, 201)]

right_count = 0
left_count = 0

#Main Loop
# while True:
#     ret, frame = cap.read()

#     # Stop when the video ends
#     if not ret:
#         print("Video playback finished.")
#         break

#     frame = cv2.resize(frame, (1280, 720))

#     frame = cv2.resize(frame, (1280, 720))

# # Run YOLO on the clean frame.
#     result = model.track(
#     frame,
#     persist=True,
#     device=0,
#     half=True,
#     classes=[2, 3, 5, 7],
#     tracker="bytetrack.yaml",
#     verbose=False)

#     # Draw overlays after inference.
#     frame = draw_roi(frame, left_side_points)
#     frame = draw_roi(frame, right_side_points)
#     frame = draw_line(frame, left_line_points)
#     frame = draw_line(frame, right_line_points) 

#     cv2.putText(frame, f'Left : {left_count}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
#     cv2.putText(frame, f'Right : {right_count}', (900,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)


#     # Display the original video frame
#     cv2.imshow("Video Playback", frame)

#     # Press Q to exit
#     if cv2.waitKey(25) & 0xFF == ord("q"):
#         print("Video playback stopped by user.")
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()

# print("Video closed successfully.")

frame_index = 0

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video playback finished.")
        break

    # Use the source video's timeline, not wall-clock inference time.
    current_video_time = frame_index / fps
    frame_index += 1

    # Resize the frame to match the ROI and line coordinates.
    frame = cv2.resize(frame, (1280, 720))

    # Run tracking on the clean frame.
    results = model.track(
        frame,
        persist=True,
        device=0,
        half=True,
        classes=[2, 3, 5, 7],
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Draw ROI overlays after inference.
    frame = draw_roi(frame, left_side_points)
    frame = draw_roi(frame, right_side_points)

    # Draw the counting lines.
    frame = draw_line(frame, left_line_points)
    frame = draw_line(frame, right_line_points)

    # Detect crossings, draw boxes, and update counters.
    frame, left_count, right_count = draw_bbox(
        frame,
        results,
        left_side_points,
        right_side_points,
        left_line_points,
        right_line_points,
        left_count,
        right_count,
        current_video_time
    )

    # Display counters.
    cv2.putText(
        frame,
        f"Left: {left_count}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )

    cv2.putText(
        frame,
        f"Right: {right_count}",
        (900, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3
    )

    # Rolling traffic flow: crossings in the latest 15 seconds, converted
    # to a per-minute rate. Left and right directions are calculated separately.
    left_flow_vpm, left_window_count = update_rolling_flow(
        left_crossing_times, current_video_time
    )
    right_flow_vpm, right_window_count = update_rolling_flow(
        right_crossing_times, current_video_time
    )

    left_flow_status = classify_flow(left_flow_vpm)
    right_flow_status = classify_flow(right_flow_vpm)


    # Separate congestion estimate: ROI occupancy plus apparent track movement.
    left_density, left_movement, left_congestion = estimate_roi_congestion(
        left_side_points, LEFT_DENSITY_MODERATE, LEFT_DENSITY_HIGH, current_video_time
    )
    right_density, right_movement, right_congestion = estimate_roi_congestion(
        right_side_points, RIGHT_DENSITY_MODERATE, RIGHT_DENSITY_HIGH, current_video_time
    )

    severity = {
        "LOW CONGESTION": 0, "SLOW / QUEUED": 1, "MODERATE DENSITY": 2,
        "MODERATE CONGESTION": 3, "HIGH DENSITY": 4,
        "HIGH CONGESTION": 5, "SEVERE CONGESTION": 6,
    }
    overall_congestion = max(
        (left_congestion, right_congestion),
        key=lambda label: severity.get(label, 0)
    )

    # Clear, non-duplicated dashboard: cumulative counts + current ROI occupancy
    # + congestion status. Flow-rate text is intentionally not displayed.
    cv2.putText(
        frame, f"TRAFFIC CONDITION: {overall_congestion}",
        (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
        (0, 255, 255), 2, cv2.LINE_AA
    )
    draw_congestion_panel(
        frame, "LEFT ROI", left_density, left_movement, left_congestion, 140
    )
    draw_congestion_panel(
        frame, "RIGHT ROI", right_density, right_movement, right_congestion, 175
    )

    out.write(frame)

    cv2.imshow("Video Playback", frame)

    if cv2.waitKey(25) & 0xFF == ord("q"):
        print("Video playback stopped by user.")
        break

# Release video resources
# Release video resources
cap.release()
out.release()
cv2.destroyAllWindows()

print("\nVideo processing completed!")
print("Final left count:", left_count)
print("Final right count:", right_count)
print("Saved video:", output_path)
print("Video closed successfully.")
