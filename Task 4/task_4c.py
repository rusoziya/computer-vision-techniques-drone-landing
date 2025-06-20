import cv2
import numpy as np
from math import asin, degrees, sqrt
import matplotlib.pyplot as plt

# ----------------------------
# Parameters and Camera Setup
# ----------------------------
video_path = "./dataset/SideView.mp4"
angular_velocity = 0.015  # rad/s
distance_to_sphere = 11.0  # meters
REPORT_INTERVAL = 100  # Record metrics every 100 frames

# Camera matrix (provided)
camera_matrix = np.array([[916.09417071,   0.0,         717.48350614],
                          [  0.0,         916.49448141, 356.7785026],
                          [  0.0,           0.0,           1.0]], dtype=np.float64)
dist_coeffs = np.array([0.01767782, 0.06540775, -0.00077648, -0.00068277, -0.24680632], dtype=np.float64)

# Extract focal length (assuming fx ~ fy)
fx = camera_matrix[0, 0]

# Optical flow parameters
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Hough circle detection parameters
hough_params = {
    'hough_param1': 225,
    'hough_param2': 46,
    'dp': 1.0,
    'minDist': 1500,
    'minRadius': 100,
    'maxRadius': 959
}

def combined_segmentation(image, **kwargs):
    """
    Detects circles in the provided image using Hough Circle Transform.
    """
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

# ----------------------------
# Open Video and Detect Sphere on First Frame
# ----------------------------
video = cv2.VideoCapture(video_path)
if not video.isOpened():
    print("Error: Unable to load video.")
    exit()

ret, frame0 = video.read()
if not ret:
    print("Error: Unable to read the first frame from the video.")
    exit()

circles = combined_segmentation(frame0, **hough_params)
if circles is None or len(circles[0]) == 0:
    print("No circular bounding found in the first frame.")
    exit()

cx0, cy0, r0 = map(int, circles[0][0])

# Define six tracking points along the vertical axis of the sphere
track_points = np.array([
    [cx0, cy0 - 0.6 * r0],
    [cx0, cy0 + 0.6 * r0],
    [cx0, cy0 - 0.1 * r0],
    [cx0, cy0 + 0.1 * r0],
    [cx0, cy0 - 0.9 * r0],
    [cx0, cy0 + 0.9 * r0]
], dtype=np.float32).reshape(-1, 1, 2)

prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
p0 = track_points.copy()

fps = video.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Default to 30 if FPS is not available
frame_count = 0
initial_positions = p0.copy()
num_points = len(p0)  # should be 6

# Time limit after which if any point is lost, we stop
time_limit_frames = int(2 * fps)  # 2 seconds of tracking

# ----------------------------
# Prepare Output File
# ----------------------------
output_filename = "metrics_output.txt"
with open(output_filename, 'w') as f:
    f.write("Frame\tPointIndex\tX_pix\tY_pix\tLatitude_deg\tPix_Displacement\tPix_Velocity(px/s)\t"
            "Theoretical_Velocity(px/s)\tReal_Displacement(m)\tReal_Velocity(m/s)\tReal_Theoretical_Velocity(m/s)\n")

# ----------------------------
# Data Storage for Plotting
# ----------------------------
frames_list = []
px_velocities = [[] for _ in range(num_points)]
m_velocities = [[] for _ in range(num_points)]
latitudes = [[] for _ in range(num_points)]
real_velocities_for_latitude = [[] for _ in range(num_points)]

# ----------------------------
# Define Smoothing Function
# ----------------------------
def moving_average(data, window_size=5):
    """
    Applies a simple moving average to the data.
    """
    if len(data) < window_size:
        return []
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ----------------------------
# Define Point Information for Annotation
# ----------------------------
# List of tuples containing point name and its distance from center (in units of r0)
point_info = [
    ("Point 0", "0.6*r0"),
    ("Point 1", "0.6*r0"),
    ("Point 2", "0.1*r0"),
    ("Point 3", "0.1*r0"),
    ("Point 4", "0.9*r0"),
    ("Point 5", "0.9*r0")
]

# ----------------------------
# Process Video Frames
# ----------------------------
try:
    while True:
        ret, frame = video.read()
        if not ret:
            print("End of video reached.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        p1, st, err = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None, **lk_params)
        good_new = p1[st == 1]

        # Check if we lost any point after 2 seconds of initialization
        if frame_count > time_limit_frames and len(good_new) < num_points:
            print("A point was lost after 2 seconds of initialization. Stopping.")
            break

        # Update p0 only if we still have all points (after time_limit_frames)
        if len(good_new) == num_points:
            p0 = good_new.reshape(-1, 1, 2)
            prev_gray = gray.copy()
        else:
            # If points are lost before time_limit_frames, keep p0 unchanged
            pass

        # Every REPORT_INTERVAL frames, compute and record metrics
        if frame_count > 0 and frame_count % REPORT_INTERVAL == 0 and len(good_new) == num_points:
            time_interval = REPORT_INTERVAL / fps
            with open(output_filename, 'a') as f:
                # Record the frame count
                frames_list.append(frame_count)
                for i, (initial_pt, current_pt) in enumerate(zip(initial_positions, p0)):
                    ix, iy = initial_pt.ravel()
                    cx, cy = current_pt.ravel()

                    dx = cx - ix
                    dy = cy - iy
                    displacement_pixels = sqrt(dx**2 + dy**2)
                    velocity_pixels = displacement_pixels / time_interval  # px/s

                    # Latitude based on initial vertical offset
                    initial_offset = (iy - cy0)
                    try:
                        ratio = initial_offset / r0
                        # Clamp the ratio to [-1, 1] to avoid domain error in asin
                        ratio = max(min(ratio, 1.0), -1.0)
                        latitude_radians = asin(ratio)
                        latitude_degrees = degrees(latitude_radians)
                    except ValueError:
                        # Handle potential domain error
                        latitude_degrees = 0.0

                    # Theoretical pixel velocity
                    try:
                        horizontal_radius_px = sqrt(r0**2 - (initial_offset**2))
                        theoretical_velocity_px = angular_velocity * horizontal_radius_px  # px/s
                    except ValueError:
                        theoretical_velocity_px = 0.0  # If sqrt of negative number

                    # Convert pixels to meters
                    displacement_m = displacement_pixels * (distance_to_sphere / fx)
                    velocity_m_s = velocity_pixels * (distance_to_sphere / fx)
                    # Theoretical in meters/s
                    try:
                        horizontal_radius_m = horizontal_radius_px * (distance_to_sphere / fx)
                        theoretical_velocity_m_s = angular_velocity * horizontal_radius_m
                    except:
                        theoretical_velocity_m_s = 0.0

                    # Write to file
                    f.write(f"{frame_count}\t{i}\t{cx:.4f}\t{cy:.4f}\t{latitude_degrees:.4f}\t"
                            f"{displacement_pixels:.4f}\t{velocity_pixels:.4f}\t"
                            f"{theoretical_velocity_px:.4f}\t{displacement_m:.4f}\t"
                            f"{velocity_m_s:.4f}\t{theoretical_velocity_m_s:.4f}\n")

                    # Store in arrays for plotting
                    px_velocities[i].append(velocity_pixels)
                    m_velocities[i].append(velocity_m_s)
                    latitudes[i].append(latitude_degrees)
                    real_velocities_for_latitude[i].append(velocity_m_s)

            # Reset initial positions
            initial_positions = p0.copy()

        # Visualization of tracking points
        for pt in p0:
            x, y = pt.ravel()
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green dot
        cv2.circle(frame, (cx0, cy0), r0, (0, 255, 0), 2)  # Green circle

        cv2.imshow('Tracking', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Break signal received. Exiting loop.")
            break

        frame_count += 1

except KeyboardInterrupt:
    # Handle any unexpected interruptions gracefully
    print("Interrupted by user.")

finally:
    # Release resources
    video.release()
    cv2.destroyAllWindows()

    print(f"Metrics saved to {output_filename}")

    # ----------------------------
    # Plotting
    # ----------------------------
    if len(frames_list) > 0:
        # ----------------------------
        # First Task: Smoothing Velocity vs Frame
        # ----------------------------
        # Apply moving average to Pixel Velocities
        window_size = 5  # Define window size for moving average
        smoothed_px_velocities = [moving_average(px_velocities[i], window_size) for i in range(num_points)]
        smoothed_m_velocities = [moving_average(m_velocities[i], window_size) for i in range(num_points)]

        # Adjust frames_list for smoothed data
        smoothed_frames = [frames_list[i] for i in range(len(frames_list)) if i >= window_size - 1]

        # Plot Pixel Velocity vs Frame (Raw)
        plt.figure(figsize=(10, 6))
        for i in range(num_points):
            if len(px_velocities[i]) == len(frames_list):
                label = f"{point_info[i][0]} ({point_info[i][1]}) Raw"
                plt.plot(frames_list, px_velocities[i], marker='o', label=label)
        plt.title("Pixel Velocity vs Frame (Raw)", fontsize=16)
        plt.xlabel("Frame", fontsize=14)
        plt.ylabel("Velocity (px/s)", fontsize=14)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.savefig("pixel_velocity_vs_frame_raw.png")
        plt.show()

        # Plot Pixel Velocity vs Frame (Smoothed)
        plt.figure(figsize=(10, 6))
        for i in range(num_points):
            if len(smoothed_px_velocities[i]) > 0:
                label = f"{point_info[i][0]} ({point_info[i][1]}) Smoothed"
                plt.plot(smoothed_frames, smoothed_px_velocities[i], marker='o', label=label)
        plt.title("Pixel Velocity vs Frame (Smoothed)", fontsize=16)
        plt.xlabel("Frame", fontsize=14)
        plt.ylabel("Velocity (px/s)", fontsize=14)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.savefig("pixel_velocity_vs_frame_smoothed.png")
        plt.show()

        # Plot Real Velocity vs Frame (Raw)
        plt.figure(figsize=(10, 6))
        for i in range(num_points):
            if len(m_velocities[i]) == len(frames_list):
                label = f"{point_info[i][0]} ({point_info[i][1]}) Raw"
                plt.plot(frames_list, m_velocities[i], marker='o', label=label)
        plt.title("Real Velocity vs Frame (Raw)", fontsize=16)
        plt.xlabel("Frame", fontsize=14)
        plt.ylabel("Velocity (m/s)", fontsize=14)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.savefig("real_velocity_vs_frame_raw.png")
        plt.show()

        # Plot Real Velocity vs Frame (Smoothed)
        plt.figure(figsize=(10, 6))
        for i in range(num_points):
            if len(smoothed_m_velocities[i]) > 0:
                label = f"{point_info[i][0]} ({point_info[i][1]}) Smoothed"
                plt.plot(smoothed_frames, smoothed_m_velocities[i], marker='o', label=label)
        plt.title("Real Velocity vs Frame (Smoothed)", fontsize=16)
        plt.xlabel("Frame", fontsize=14)
        plt.ylabel("Velocity (m/s)", fontsize=14)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.savefig("real_velocity_vs_frame_smoothed.png")
        plt.show()

        # ----------------------------
        # Second Task: Polynomial Fit on Mean Real Velocities
        # ----------------------------
        # Calculate mean real velocity and mean latitude for each point
        mean_real_velocities = []
        mean_latitudes = []
        for i in range(num_points):
            if len(real_velocities_for_latitude[i]) > 0 and len(latitudes[i]) > 0:
                mean_velocity = np.mean(real_velocities_for_latitude[i])
                mean_latitude = np.mean(latitudes[i])
                mean_real_velocities.append(mean_velocity)
                mean_latitudes.append(mean_latitude)
            else:
                mean_real_velocities.append(0.0)
                mean_latitudes.append(0.0)

        # Fit a second-degree polynomial through the six mean points
        # A second-degree polynomial (quadratic) is suitable for capturing parabolic trends
        if len(mean_latitudes) == num_points:
            coeffs = np.polyfit(mean_latitudes, mean_real_velocities, deg=2)
            poly_fit = np.poly1d(coeffs)

            # Generate x values for plotting the polynomial
            lat_min = min(mean_latitudes) - 5
            lat_max = max(mean_latitudes) + 5
            lat_range = np.linspace(lat_min, lat_max, 300)
            poly_values = poly_fit(lat_range)

        # ----------------------------
        # Plot Latitude vs Real Velocity with Polynomial Fit
        # ----------------------------
        plt.figure(figsize=(10, 6))
        # Scatter plot of all real velocities vs latitude
        for i in range(num_points):
            if len(latitudes[i]) > 0 and len(real_velocities_for_latitude[i]) > 0:
                label = f"{point_info[i][0]} ({point_info[i][1]}) Data"
                plt.scatter(latitudes[i], real_velocities_for_latitude[i], label=label)
        # Scatter plot of mean points
        plt.scatter(mean_latitudes, mean_real_velocities, color='red', label='Mean Velocities')
        # Plot polynomial fit
        if len(mean_latitudes) == num_points:
            plt.plot(lat_range, poly_values, color='black', linestyle='--', label='Polynomial Fit (Degree 2)')
        plt.title("Latitude vs Real Velocity with Polynomial Fit (Mean)", fontsize=16)
        plt.xlabel("Latitude (degrees)", fontsize=14)
        plt.ylabel("Real Velocity (m/s)", fontsize=14)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.savefig("latitude_vs_real_velocity_with_fit_mean.png")
        plt.show()

        # ----------------------------
        # Plot Latitude vs Real Velocity (Raw Data)
        # ----------------------------
        plt.figure(figsize=(10, 6))
        for i in range(num_points):
            if len(latitudes[i]) > 0 and len(real_velocities_for_latitude[i]) > 0:
                label = f"{point_info[i][0]} ({point_info[i][1]})"
                plt.scatter(latitudes[i], real_velocities_for_latitude[i], label=label)
        plt.title("Latitude vs Real Velocity (Raw Data)", fontsize=16)
        plt.xlabel("Latitude (degrees)", fontsize=14)
        plt.ylabel("Real Velocity (m/s)", fontsize=14)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)
        plt.savefig("latitude_vs_real_velocity_raw.png")
        plt.show()

    else:
        print("No data available to plot.")


#Theoretical calculations based on ground truths for reference
R_actual = 3.5  # Known reference or calculated From Task 4a, in meters
T_actual = 490  # From Task 3c for SideView, in seconds

# Compute angular velocity
omega = 2 * np.pi / T_actual

# Latitude values in degrees
latitudes = np.linspace(-90, 90, 100)  # 100 points between -90° and 90°
latitudes_rad = np.radians(latitudes)  # Convert to radians

# Compute linear velocity
r_phi = R_actual * np.cos(latitudes_rad)  # Radius at each latitude
v_phi = omega * r_phi  # Linear velocity

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(latitudes, v_phi)
plt.title("Surface Linear Velocity vs Latitude (Theory)", fontsize=16)
plt.xlabel("Latitude (degrees)", fontsize=14)
plt.ylabel("Real Velocity (m/s)", fontsize=14)
plt.legend(fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.savefig("latitude_vs_real_velocity_theory.png")
plt.show()

