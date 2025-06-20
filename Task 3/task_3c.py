import cv2
import numpy as np

# Path to captured video
video_path = "./dataset/FrontView1.mp4"

video = cv2.VideoCapture(video_path)

# Ensure the video is loaded properly
if not video.isOpened():
    print("Error: Unable to load video.")
    exit()

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
    'maxRadius': 1500
}

# Segmentation function
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

while True:
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

        matched_frame = cv2.drawMatches(roi_frame0, kp0, roi_frame, kp, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        geometric_accuracy = geometric_consistency_accuracy(kp0, kp, matches)
        if geometric_accuracy > 85.0:
            print(f"High Accuracy Match at Frame {frame_count}: {geometric_accuracy:.2f}%")
            rotation_frames.append((frame_count, geometric_accuracy))

        cv2.circle(frame, (cx, cy), r, (0, 255, 0), 2)

        # Resize frames to specified dimensions
        circular_bounding_resized = cv2.resize(frame, (640, 400))
        matched_frame_resized = cv2.resize(matched_frame, (800, 400))

        # Combine the resized frames horizontally
        combined_view = np.hstack((circular_bounding_resized, matched_frame_resized))

        # Add text for frame count and accuracy on the combined view with a solid background
        text_frame = f"Frame: {frame_count}"
        text_accuracy = f"Match: {geometric_accuracy:.2f}%"

        # Choose position and font details for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Smaller text scale
        font_thickness = 1
        text_color = (0, 0, 0)  # Black text color
        background_color = (255, 255, 255)  # White background rectangle

        # Calculate the size of the text background (ensure both texts have matching rectangles)
        text_frame_size = cv2.getTextSize(text_frame, font, font_scale, font_thickness)[0]
        text_accuracy_size = cv2.getTextSize(text_accuracy, font, font_scale, font_thickness)[0]

        # Set the position of the text (top-left corner)
        text_frame_position = (10, 25)
        text_accuracy_position = (10, 45)

        # Define the maximum width to ensure both rectangles match
        max_width = max(text_frame_size[0], text_accuracy_size[0])

        # Draw filled rectangles as background for the text
        cv2.rectangle(combined_view, 
                    (text_frame_position[0] - 5, text_frame_position[1] - 20),
                    (text_frame_position[0] + max_width + 5, text_frame_position[1] + 5),
                    background_color, -1)

        cv2.rectangle(combined_view, 
                    (text_accuracy_position[0] - 5, text_accuracy_position[1] - 20),
                    (text_accuracy_position[0] + max_width + 5, text_accuracy_position[1] + 5),
                    background_color, -1)

        # Add the text on top of the background rectangles
        cv2.putText(combined_view, text_frame, text_frame_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(combined_view, text_accuracy, text_accuracy_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        # Show combined view
        cv2.imshow('Combined View', combined_view)

        # Save screenshot if 's' is pressed
        if cv2.waitKey(1) & 0xFF == ord('s'):
            screenshot_path = f"screenshot_frame_{frame_count}.png"
            cv2.imwrite(screenshot_path, combined_view)
            print(f"Screenshot saved: {screenshot_path}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += frame_step

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
    fps = 30  # Example FPS
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
