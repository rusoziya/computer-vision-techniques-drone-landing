import cv2
import numpy as np

# Path to captured video
video_path = "./dataset/FrontView1.mp4"

video = cv2.VideoCapture(video_path)

# Ensure the video is loaded properly
if not video.isOpened():
    print("Error: Unable to load video.")
    exit()

# Read the first frame to define ROI
ret, frame0 = video.read()
if not ret:
    print("Error: Unable to read the first frame from the video.")
    exit()

# Let the user define an ROI on the first frame
roi = cv2.selectROI("Select ROI on First Frame", frame0, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Crop the ROI from the first frame
roi_frame0 = frame0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
# print(roi_frame0)

# Initialize ORB detector
orb = cv2.ORB_create()

# Detect and compute features in the ROI of the first frame
kp0, des0 = orb.detectAndCompute(roi_frame0, None)

# Define accuracy calculation functions

# Not used
def calculate_match_accuracy(matches, distance_threshold=50):
    """Calculate the percentage of good matches under a distance threshold."""
    good_matches = [m for m in matches if m.distance < distance_threshold]
    accuracy = len(good_matches) / len(matches) * 100 if matches else 0
    return accuracy


def geometric_consistency_accuracy(kp1, kp2, matches):
    """Check spatial/geometric consistency using fundamental matrix estimation."""
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
    inlier_matches = mask.ravel().tolist() if mask is not None else []
    accuracy = sum(inlier_matches) / len(matches) * 100 if matches else 0
    return accuracy


# Set the number of frames to skip
frame_step = 100 # You can change this value to process fewer/more frames
frame_rate = video.get(cv2.CAP_PROP_FPS)  # Frame rate of the video

frame_count = 0  # Initialize frame counter
rotation_frames = []  # To store frames with high geometric accuracy

while True:
    # Skip to the desired frame using set() with CAP_PROP_POS_FRAMES
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
    
    # Read the current frame
    ret, frame = video.read()
    if not ret:
        print("End of video reached or cannot read frame.")
        break

    # Crop the same ROI from the current frame
    roi_frame = frame[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]

    # Detect and compute ORB features in the cropped ROI
    kp, des = orb.detectAndCompute(roi_frame, None)

    # Match features only if descriptors are found
    if des is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des0, des)

        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Calculate accuracy metrics
        geometric_accuracy = geometric_consistency_accuracy(kp0, kp, matches)
        
        if geometric_accuracy > 85.0:  # High accuracy threshold
            print(f"High Accuracy Match at Frame {frame_count}: {geometric_accuracy:.2f}%")
            rotation_frames.append((frame_count, geometric_accuracy))

    # Skip the desired number of frames
    frame_count += frame_step

# Cleanup resources
video.release()
cv2.destroyAllWindows()

# Calculate rotation time
if rotation_frames:
    # Define a threshold for clustering (e.g., 500 frames)
    frame_gap_threshold = 500

    # Step 1: Group frames into clusters
    clusters = []
    current_cluster = [rotation_frames[0]]

    for i in range(1, len(rotation_frames)):
        frame_idx, accuracy = rotation_frames[i]
        prev_frame_idx, _ = rotation_frames[i - 1]

        if frame_idx - prev_frame_idx <= frame_gap_threshold:
            current_cluster.append(rotation_frames[i])
        else:
            clusters.append(current_cluster)
            current_cluster = [rotation_frames[i]]

    # Add the last cluster
    clusters.append(current_cluster)

    # Step 2: Select the best match (highest accuracy) from each cluster
    selected_frames = [
        max(cluster, key=lambda x: x[1]) for cluster in clusters
    ]

    # Step 3: Estimate rotation times
    fps = 30
    rotation_times = []
    for i in range(1, len(selected_frames)):
        frame_diff = selected_frames[i][0] - selected_frames[i - 1][0]
        rotation_times.append(frame_diff / fps)

    # Compute average rotation time
    if rotation_times:
        avg_rotation_time = np.mean(rotation_times)
        print("High Accuracy Frames:", selected_frames)
        print("Rotation Times:", rotation_times)
        print(f"Average Rotation Time: {avg_rotation_time:.2f} seconds")
    else:
        print("Not enough data for rotation time estimation.")

else:
    print("Not enough matches found to estimate rotation time.")
