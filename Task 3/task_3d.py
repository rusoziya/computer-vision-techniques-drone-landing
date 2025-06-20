import cv2
import numpy as np
import logging
import os
from datetime import datetime
import time

# Path to captured video
video_path = "./dataset/FrontView1.mp4"

video = cv2.VideoCapture(video_path)

# Ensure the video is loaded properly
if not video.isOpened():
    print("Error: Unable to load video.")
    exit()

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, f"processing_times_{timestamp}.log")

logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Read the first frame to define ROI using segmentation
ret, frame0 = video.read()
if not ret:
    print("Error: Unable to read the first frame from the video.")
    exit()

# Define segmentation parameters
params = {
    'hough_param1': 225,
    'hough_param2': 46,
    'dp': 1.0,
    'minDist': 1500,
    'minRadius': 100,
    'maxRadius': 959
}

# Circular segmentation function
def segmentation(image, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=kwargs['dp'],
        minDist=kwargs['minDist'],
        param1=kwargs['hough_param1'],
        param2=kwargs['hough_param2'],
        minRadius=kwargs['minRadius'],
        maxRadius=kwargs['maxRadius']
    )
    return circles

# Extract circular bounding ROI
def get_circular_roi(image, circle):
    cx, cy, r = map(int, circle)
    x1, y1 = max(0, cx - r), max(0, cy - r)
    x2, y2 = min(image.shape[1], cx + r), min(image.shape[0], cy + r)
    mask = np.zeros_like(image, dtype=np.uint8)
    cv2.circle(mask, (cx, cy), r, (255, 255, 255), -1)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image[y1:y2, x1:x2], (cx, cy, r)

# Calculate geometric accuracy
def geometric_consistency_accuracy(kp1, kp2, matches):
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    inlier_matches = mask.ravel().tolist() if mask is not None else []
    accuracy = sum(inlier_matches) / len(matches) * 100 if matches else 0
    return accuracy

# Process first frame
circles = segmentation(frame0, **params)
if circles is None or len(circles[0]) == 0:
    print("No circular bounding found in the first frame.")
    exit()

roi_frame0, circle0 = get_circular_roi(frame0, circles[0][0])
orb = cv2.ORB_create()
kp0, des0 = orb.detectAndCompute(roi_frame0, None)

# Frame processing
frame_step = 100
frame_rate = video.get(cv2.CAP_PROP_FPS)
frame_count = 0
rotation_frames = []
clusters = []
cluster_threshold = 2000  # Maximum frame gap for clustering
rotation_time = "Calculating..."
processing_times = []

while True:
    start_time = time.time()  # Start timing

    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    ret, frame = video.read()
    if not ret:
        break

    circles = segmentation(frame, **params)
    if circles is None or len(circles[0]) == 0:
        print(f"No circular bounding found at frame {frame_count}. Skipping.")
        frame_count += frame_step
        continue

    roi_frame, circle = get_circular_roi(frame, circles[0][0])
    cx, cy, r = map(int, circle)

    kp, des = orb.detectAndCompute(roi_frame, None)
    if des is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des0, des)
        matches = sorted(matches, key=lambda x: x.distance)

        geometric_accuracy = geometric_consistency_accuracy(kp0, kp, matches)
        if geometric_accuracy > 85.0:
            logging.info(f"High Accuracy Match at Frame {frame_count}: {geometric_accuracy:.2f}%")
            print(f"High Accuracy Match at Frame {frame_count}: {geometric_accuracy:.2f}%")
            rotation_frames.append((frame_count, geometric_accuracy))

            # Handle clustering dynamically
            if not clusters or frame_count - clusters[-1][-1][0] > cluster_threshold:
                # Start a new cluster
                clusters.append([(frame_count, geometric_accuracy)])
            else:
                # Append to the last cluster
                clusters[-1].append((frame_count, geometric_accuracy))

            # If a new cluster forms, compute rotation time
            if len(clusters) > 1:
                prev_cluster = clusters[-2]
                current_cluster = clusters[-1]
                best_prev_frame = max(prev_cluster, key=lambda x: x[1])[0]
                best_current_frame = max(current_cluster, key=lambda x: x[1])[0]
                rotation_time = (best_current_frame - best_prev_frame) / frame_rate

        if (frame_count % 500 == 0):
            if isinstance(rotation_time, str):
                print("Estimated Rotation Time: Calculating...")
            else:
                print(f"Estimated Rotation Time: {rotation_time:.2f} seconds")
    
    # Measure processing time
    processing_time = time.time() - start_time
    processing_times.append(processing_time)
    avg_processing_time = np.mean(processing_times)
    avg_fps = 1/avg_processing_time

    # Log to file
    logging.info(f"Frame: {frame_count}, Processing Time: {processing_time:.4f}, Est. Rotation Time: {rotation_time}, Accuracy: {geometric_accuracy:.2f}%")
    if (frame_count % 500 == 0):
        logging.info(f"Average processing time per frame at this point: {avg_processing_time:.4f} seconds or {avg_fps:.2f} FPS")
    frame_count += frame_step

video.release()
cv2.destroyAllWindows()
