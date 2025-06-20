import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt
from task2_segment_funct import combined_segmentation_hough


# Find centroids function
def find_points(final_mask):
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        print("No contours found in the mask.")
        return None, None, None
    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    center_x = int(x)
    center_y = int(y)
    radius = int(radius)

    return center_x, center_y, radius


# Process the video
def analyze_video(video_path, output_dir, seconds=60):
    centers = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0
    success, frame = cap.read()
    
    central_frame = None
    central_frame_index = None
    
    while success and frame_count < seconds * frame_rate:  # Process up to 60 seconds
        if frame_count % frame_rate == 0:  # Take a frame every second
            frame_path = f"{output_dir}/frame_{frame_count // frame_rate}.png"
            cv2.imwrite(frame_path, frame) 
            final_mask, segmented_image, _ = combined_segmentation_hough(
                    frame_path)
            if final_mask is not None:
                center_x, center_y, radius  = find_points(final_mask)
                if center_x is not None:
                    centers.append((center_x, center_y))
                    if len(centers) == 30:  # Capture the middle frame
                        central_frame = frame
                        central_frame_index = frame_count // frame_rate
                        print(f"Radius: {radius} px")
        frame_count += 1
        success, frame = cap.read()
    
    cap.release()
    return centers, central_frame, central_frame_index


# Visualize the centroids on a frame
def visualize_centers_on_frame(frame, centers, central_index, mask_draw=False):
    frame_with_centers = frame.copy()

    # Initialize the distances to None
    left_to_right_distance = None
    left_to_center_distance = None
    right_to_center_distance = None

    # Identify the most left, most right, and center points
    if len(centers) > 0:
        most_left = min(centers, key=lambda x: x[0])  # Point with smallest x value
        most_right = max(centers, key=lambda x: x[0])  # Point with largest x value
        center_point = centers[central_index]  # Central point based on the given index

        # Plot three masks with these centroids and their perpendicular axis
        # Draw the masks on the frame
        if mask_draw is True:
            masks = []
            for i, center in enumerate([most_left, most_right, center_point]):
                # Create a blank mask
                mask = np.zeros_like(frame, dtype=np.uint8)

                # Draw the centroid as a circle
                # Draw the white circle
                cv2.circle(mask, (center[0], center[1]), 200, (255, 255, 255), -1)

                # Draw the centroid as a red dot
                if i == 0:
                    cv2.circle(mask, (center[0], center[1]), 9, (0, 0, 255), -1)
                    cv2.putText(mask, f"({center[0]},{center[1]})", (center[0] + 15, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                if i == 1:
                    cv2.circle(mask, (center[0], center[1]), 9, (0, 255, 0), -1)
                    cv2.putText(mask, f"({center[0]},{center[1]})", (center[0] + 15, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
                if i == 2:
                    cv2.circle(mask, (center[0], center[1]), 9, (255, 0, 0), -1)
                    cv2.putText(mask, f"({center[0]},{center[1]})", (center[0] + 15, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 3)

                # Save the mask for further use
                masks.append(mask)

                # Save the mask as an image file
                mask_filename = f"Results_Images/task2.2/mask_{['most_left', 'most_right', 'center'][i]}.png"
                cv2.imwrite(mask_filename, mask)

                # Add text to the mask for visualization (optional)
                cv2.putText(mask, f"({center[0]},{center[1]})", (center[0] + 25, center[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Optionally display the mask
                plt.figure(figsize=(5, 5))
                plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
                plt.title(f"Mask for {['Most Left', 'Most Right', 'Center'][i]} Point")
                plt.axis("off")
                plt.show()

        # Calculate the Euclidean distances between the points
        left_to_right_distance = np.sqrt((most_right[0] - most_left[0])**2 + (most_right[1] - most_left[1])**2)
        left_to_center_distance = np.sqrt((center_point[0] - most_left[0])**2 + (center_point[1] - most_left[1])**2)
        right_to_center_distance = np.sqrt((center_point[0] - most_right[0])**2 + (center_point[1] - most_right[1])**2)

        # Write these distance on a txt file
        with open(f"Results_Images/task2.2/NEW_Results.txt", "a") as f:
            f.write(f"Results for {mins} minutes\n")
            f.write(f"Abs Distance between Most Left and Most Right: {left_to_right_distance:.2f} pixels\n")
            f.write(f"Abs Distance between Most Left and Center: {left_to_center_distance:.2f} pixels\n")
            f.write(f"Abs Distance between Most Right and Center: {right_to_center_distance:.2f} pixels\n")
            f.write(f"x Displacement between Most Left and Most Right: {most_right[0] - most_left[0]:.2f} pixels\n")
            f.write(f"x Displacement between Most Left and Center: {center_point[0] - most_left[0]:.2f} pixels\n")
            f.write(f"x Displacement between Most Right and Center: {center_point[0] - most_right[0]:.2f} pixels\n")

        # Plot the points on the frame
        cv2.circle(frame_with_centers, (most_left[0], most_left[1]), 7, (255, 0, 0), -1)  # Blue for leftmost
        cv2.putText(frame_with_centers, "ML", (most_left[0]-35, most_left[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(frame_with_centers, (most_right[0], most_right[1]), 7, (0, 0, 255), -1)  # Red for rightmost
        cv2.putText(frame_with_centers,"  MR", (most_right[0], most_right[1]),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.circle(frame_with_centers, (center_point[0], center_point[1]), 7, (0, 255, 0), -1)  # Green for center
        cv2.putText(frame_with_centers, "EQ", (center_point[0], center_point[1]+18),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw dashed lines (short lines) from the center to the other points
        # For dashed line, draw a series of small lines
        line_thickness = 1
        dash_length = 5  # Length of each dash
        dash_gap = 5  # Gap between dashes

        def draw_dashed_line(image, point1, point2, color, thickness=1):
            # Calculate the distance between the points
            total_length = int(np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2))
            num_dashes = total_length // (dash_length + dash_gap)

            # Draw the dashed line
            for i in range(num_dashes):
                start_x = int(point1[0] + i * (point2[0] - point1[0]) / num_dashes)
                start_y = int(point1[1] + i * (point2[1] - point1[1]) / num_dashes)
                end_x = int(point1[0] + (i + 1) * (point2[0] - point1[0]) / num_dashes)
                end_y = int(point1[1] + (i + 1) * (point2[1] - point1[1]) / num_dashes)
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, thickness)

        # Draw dashed lines from the center point to the leftmost and rightmost points
        draw_dashed_line(frame_with_centers, center_point, most_left, (255, 0, 0), thickness=1)  # Blue dashed line
        draw_dashed_line(frame_with_centers, center_point, most_right, (0, 0, 255), thickness=1)  # Red dashed line

    # Show the image
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(frame_with_centers, cv2.COLOR_BGR2RGB))
    plt.title("Most Left, Most Right, and Center Points on Frame")
    plt.axis("off")
    plt.savefig(f'Results_Images/task2.2/NEW_displac_centroids_{mins}mins.png')
    plt.show()

    # Return the computed distances for further analysis or logging
    return left_to_right_distance, left_to_center_distance, right_to_center_distance


def low_pass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def analyze_trajectory_smooth(centers):
    x_vals = [x for x, y in centers[45:115]]
    time = np.arange(len(x_vals))

    # Smooth the data with a low-pass filter
    fs = 1  # Sampling frequency (1 sample per second for frame-rate)
    cutoff = 0.1  # Cutoff frequency (adjust based on expected signal)
    x_vals_smooth = low_pass_filter(x_vals, cutoff, fs)

    # Fit a sine wave to smoothed x-values
    peaks, _ = find_peaks(x_vals_smooth)
    period = np.mean(np.diff(peaks)) if len(peaks) > 1 else None
    amplitude = (max(x_vals_smooth) - min(x_vals_smooth)) / 2

    with open(f"Results_Images/task2.2/NEW_Results_filter.txt", "a") as f:
        f.write(f"Period: {period} seconds\n")
        f.write(f"Amplitude: {amplitude} pixels\n")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(time, x_vals, label="X (Original Horizontal Movement)", alpha=0.5)
    plt.plot(time, x_vals_smooth, label="X (Smoothed Horizontal Movement)", linestyle='--', linewidth=2)
    #plt.plot(time, y_vals, label="Y (Original Vertical Movement)", alpha=0.5)
    #plt.plot(time, y_vals_smooth, label="Y (Smoothed Vertical Movement)", linestyle='--', linewidth=2)
    #plt.axhline(np.mean(x_vals_smooth), color='r', linestyle='--', label="Mean X")
    plt.title("AO Center Movement Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixel Coordinates")
    plt.legend()
    plt.grid()
    plt.savefig(f'Results_Images/task2.2/graph_smooth_x_{mins}mins.png')
    plt.show()

    return amplitude, period

def analyze_trajectory(centers):
    x_vals = [x for x, y in centers]
    y_vals = [y for x, y in centers]
    time = np.arange(len(x_vals))

    # Fit a sine wave to x-values
    peaks, _ = find_peaks(x_vals)
    period = np.mean(np.diff(peaks)) if len(peaks) > 1 else None
    amplitude = (max(x_vals) - min(x_vals)) / 2

    with open(f"Results_Images/task2.2/NEW_Results.txt", "a") as f:
            f.write(f"Period: {period} seconds\n")
            f.write(f"Amplitude: {amplitude} pixels\n")

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(time, x_vals, label="X (Horizontal Movement)")
    plt.plot(time, y_vals, label="Y (Vertical Movement)")
    plt.axhline(np.mean(x_vals), color='r', linestyle='--', label="Mean X")
    plt.title("AO Center Movement Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Pixel Coordinates")
    plt.legend(loc="center right")
    plt.grid()
    plt.savefig(f'Results_Images/task2.2/graph_x_{mins}mins.png')
    plt.show()

    return amplitude, period



# Run analysis
# Change the MINS for different trials (3,4,5)
mins = 5
video_path = "./dataset/front_view.avi"
output_dir = "Results_Images/task2.2/frames"
centers, central_frame, central_frame_index = analyze_video(video_path, output_dir, seconds=mins*60)

# Visualize centers
if central_frame is not None:
    visualize_centers_on_frame(central_frame, centers, central_frame_index, mask_draw=False)

# Analyze trajectory (original)
# amplitude, period = analyze_trajectory(centers)
# print(f"Amplitude of swing: {amplitude} pixels")
# print(f"Period of swing: {period} seconds")

# Analyze trajectory (smoothed)
amplitude, period = analyze_trajectory_smooth(centers)
print(f"Amplitude of swing (smoothed): {amplitude} pixels")
print(f"Period of swing (smoothed): {period} seconds")
