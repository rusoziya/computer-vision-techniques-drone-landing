import pyautogui
import cv2
import numpy as np
import math
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

# Constants
REAL_DIAMETER = 7  # Diameter of Earth model in meters
ROTATION_PERIOD = 7 * 60  # 7 minutes in seconds
OMEGA = 2 * math.pi / ROTATION_PERIOD  # Angular velocity (rad/s)

# Video input
VIDEO_PATH = "./dataset/SideView.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Optical flow parameters
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Click event handler
def select_point(event, x, y, flags, param):
    global selected_point, paused, velocity_buffer
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_point = np.array([[x, y]], dtype=np.float32)
        paused = False
        velocity_buffer = []  # Clear the buffer for a fresh start

# Segmentation function
def segment_circle(image, **kwargs):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=kwargs["dp"],
        minDist=kwargs["minDist"],
        param1=kwargs["hough_param1"],
        param2=kwargs["hough_param2"],
        minRadius=kwargs["minRadius"],
        maxRadius=kwargs["maxRadius"],
    )
    return circles

# Process the first frame
ret, first_frame = cap.read()
if not ret:
    print("Error: Unable to read the first frame.")
    exit()

# Segmentation parameters
params = {
    "hough_param1": 225,
    "hough_param2": 46,
    "dp": 1.0,
    "minDist": 1500,
    "minRadius": 100,
    "maxRadius": 1500,
}
circles = segment_circle(first_frame, **params)
if circles is None or len(circles[0]) == 0:
    print("No circular bounding found in the first frame.")
    exit()

circle = circles[0][0]
cx, cy, radius_pixels = map(int, circle)
pixel_to_meter = REAL_DIAMETER / (2 * radius_pixels)

# Grayscale conversion
gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Visualization setup
cv2.namedWindow("Interactive Velocity Estimation")
cv2.setMouseCallback("Interactive Velocity Estimation", select_point)

# Initialize Matplotlib Figure
fig = Figure(figsize=(9, 6))  # Adjust the figure size (width, height in inches)
canvas = FigureCanvas(fig)
ax = fig.add_subplot(111)
time_series = []  # Store all time values
velocity_series = []  # Store all velocity values

x_limit = 40  # Static x-axis range in seconds
y_limit = 0.15
ax.set_xlim(0, x_limit)  # Set x-axis from 0 to x_limit
ax.set_ylim(0, y_limit)  # Adjust based on expected velocity range
ax.set_title("Surface Linear Velocity over Time")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity (m/s)")

# Function to update the graph
def update_plot():
    ax.clear()
    ax.plot(time_series, velocity_series, label="Linear Velocity (m/s)", color="blue")
    ax.legend()
    ax.set_xlim(0, x_limit)
    ax.set_ylim(0, y_limit)
    ax.set_title("Surface Linear Velocity over Time")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    canvas.draw()

def plot_to_image():
    """
    Renders the Matplotlib graph as a NumPy array image.
    """
    buf = canvas.buffer_rgba()
    mat_img = np.asarray(buf, dtype=np.uint8)
    mat_img = cv2.cvtColor(mat_img, cv2.COLOR_RGBA2BGR)
    return mat_img

def save_screenshot(window_name="Interactive Velocity Estimation"):
    """
    Saves a screenshot of the combined OpenCV and graph display window.
    """
    try:
        screenshot = pyautogui.screenshot()
        screenshot = np.array(screenshot)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)

        # Get the window position and size
        rect = cv2.getWindowImageRect(window_name)
        x, y, w, h = rect

        # Crop the screenshot to the window
        cropped_screenshot = screenshot[y:y + h, x:x + w]

        cv2.imwrite("visualization_screenshot.png", cropped_screenshot)
        print("Screenshot saved as 'visualization_screenshot.png'.")
    except Exception as e:
        print(f"Error capturing screenshot: {e}")

# Initialize variables
start_time = None
last_velocity = None
selected_point = None
paused = True
velocity_buffer = []
frame_counter = 0
velocity_display_interval = 10  # Update velocity display every 10 frames
path_points = []

# Main loop
while cap.isOpened():
    if paused:
        cv2.imshow("Interactive Velocity Estimation", first_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if selected_point is not None:
        next_point, st, err = cv2.calcOpticalFlowPyrLK(
            gray_first_frame, gray_frame, selected_point, None, **lk_params
        )
        if next_point is not None and st[0] == 1:
            x1, y1 = selected_point.ravel()
            x2, y2 = next_point.ravel()
            dx, dy = (x2 - x1), (y2 - y1)
            displacement_pixels = math.sqrt(dx**2 + dy**2)

            latitude = math.degrees(math.asin((y2 - cy) / radius_pixels))
            arc_length_meters = (
                displacement_pixels * pixel_to_meter * math.cos(math.radians(latitude))
            )
            time_elapsed = 1 / cap.get(cv2.CAP_PROP_FPS)
            linear_velocity = arc_length_meters / time_elapsed

            velocity_buffer.append(linear_velocity)
            if frame_counter % velocity_display_interval == 0:
                smoothed_velocity = np.mean(velocity_buffer)
                velocity_buffer = []
                last_velocity = smoothed_velocity

                if start_time is None:
                    start_time = cv2.getTickCount() / cv2.getTickFrequency()
                elapsed_time = (
                    cv2.getTickCount() / cv2.getTickFrequency() - start_time
                )
                time_series.append(elapsed_time)
                velocity_series.append(smoothed_velocity)

                update_plot()

            selected_point = next_point
            path_points.append((int(x2), int(y2)))

        gray_first_frame = gray_frame.copy()

    for i in range(1, len(path_points)):
        cv2.line(frame, path_points[i - 1], path_points[i], (0, 0, 255), 2)

    if last_velocity is not None:
        cv2.putText(
            frame,
            f"Linear Velocity: {last_velocity:.2f} m/s",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

    if selected_point is not None:
        x, y = selected_point.ravel()
        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    cv2.circle(frame, (cx, cy), radius_pixels, (255, 0, 0), 2)

    frame_counter += 1

    # Generate the graph image
    graph_image = plot_to_image()

    # Resize the graph to match the frame height
    graph_image = cv2.resize(graph_image, (frame.shape[1] // 1, frame.shape[0]))

    # Resize the frame to ensure it matches the concatenated layout
    frame_resized = cv2.resize(frame, (frame.shape[1] // 1, frame.shape[0]))

    # Combine the frame and graph horizontally
    combined_frame = cv2.hconcat([frame_resized, graph_image])

    cv2.imshow("Interactive Velocity Estimation", combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):  # Save a screenshot
        save_screenshot("Interactive Velocity Estimation")
    elif key == ord("q"):  # Quit the program
        break

cap.release()
cv2.destroyAllWindows()
