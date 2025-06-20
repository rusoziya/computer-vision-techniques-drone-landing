import cv2
import numpy as np
import sys
import os

def load_camera_parameters():
    """
    Define your camera matrix and distortion coefficients here.
    Replace the example values with your actual camera parameters.
    """
    # Example camera matrix (replace with your actual values)
    # [ [fx,  0, cx],
    #   [ 0, fy, cy],
    #   [ 0,  0,  1] ]
    camera_matrix = np.array([[916.09417071,   0.0,         717.4835064],
                              [  0.0,         916.49448141, 356.7785026],
                              [  0.0,           0.0,           1.0]], dtype=np.float64)

    # Distortion coefficients (unused now but kept for potential future use)
    dist_coeffs = np.array([0.01767782, 0.06540775, -0.00077648, -0.00068277, -0.24680632], dtype=np.float64)

    return camera_matrix, dist_coeffs

def load_mask(mask_path):
    """
    Load the segmented mask image.
    The mask should be a binary image where the planet is white (1) and the background is black (0).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Unable to load mask image from {mask_path}")
        sys.exit(1)
    # Ensure binary: planet as 1, background as 0
    _, binary_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return binary_mask

def calculate_planet_radii(mask, camera_matrix, focal_length, distance_to_planet):
    """
    Calculate the planet's radii in four directions: left, right, up, and down.

    Parameters:
    - mask: Binary mask of the planet (1 for planet, 0 for background).
    - camera_matrix: Camera matrix containing focal lengths.
    - focal_length: Focal length in pixels.
    - distance_to_planet: Distance from the camera to the planet in the same unit as desired for radius.

    Returns:
    - radii_real: Dictionary containing radii in real-world units.
    - radii_pixels: Dictionary containing radii in pixels.
    - points: Dictionary containing points defining each radius.
    - centroid: Coordinates of the centroid.
    """
    # Extract all foreground pixels
    foreground_pixels = np.column_stack(np.where(mask == 1))
    if foreground_pixels.size == 0:
        print("Error: No foreground pixels found in the mask.")
        sys.exit(1)

    # Calculate centroid of the foreground
    centroid = foreground_pixels.mean(axis=0)
    centroid_y = int(round(centroid[0]))  # y-coordinate (row)
    centroid_x = int(round(centroid[1]))  # x-coordinate (column)

    # Handle edge cases where centroid might be out of bounds
    centroid_y = max(0, min(centroid_y, mask.shape[0] - 1))
    centroid_x = max(0, min(centroid_x, mask.shape[1] - 1))

    # Function to find extreme points in a given direction
    def find_extreme_point(mask, centroid_x, centroid_y, direction, search_radius=5):
        """
        Find the extreme point in a specified direction from the centroid.

        Parameters:
        - mask: Binary mask.
        - centroid_x: Centroid x-coordinate.
        - centroid_y: Centroid y-coordinate.
        - direction: 'left', 'right', 'up', 'down'.
        - search_radius: Number of rows or columns to search if exact row/column has no pixels.

        Returns:
        - extreme_point: (x, y) tuple of the extreme point.
        """
        if direction in ['left', 'right']:
            # Horizontal directions: scan along centroid row
            x_coords = np.where(mask[centroid_y, :] == 1)[0]
            if len(x_coords) == 0:
                # Search nearby rows
                for offset in range(1, search_radius + 1):
                    if centroid_y - offset >= 0:
                        x_coords = np.where(mask[centroid_y - offset, :] == 1)[0]
                        if len(x_coords) > 0:
                            break
                    if centroid_y + offset < mask.shape[0]:
                        x_coords = np.where(mask[centroid_y + offset, :] == 1)[0]
                        if len(x_coords) > 0:
                            break
                if len(x_coords) == 0:
                    print(f"Error: No foreground pixels found near the centroid row for {direction}.")
                    sys.exit(1)
            if direction == 'left':
                extreme_x = x_coords.min()
            else:
                extreme_x = x_coords.max()
            extreme_point = (extreme_x, centroid_y)
        elif direction in ['up', 'down']:
            # Vertical directions: scan along centroid column
            y_coords = np.where(mask[:, centroid_x] == 1)[0]
            if len(y_coords) == 0:
                # Search nearby columns
                for offset in range(1, search_radius + 1):
                    if centroid_x - offset >= 0:
                        y_coords = np.where(mask[:, centroid_x - offset] == 1)[0]
                        if len(y_coords) > 0:
                            break
                    if centroid_x + offset < mask.shape[1]:
                        y_coords = np.where(mask[:, centroid_x + offset] == 1)[0]
                        if len(y_coords) > 0:
                            break
                if len(y_coords) == 0:
                    print(f"Error: No foreground pixels found near the centroid column for {direction}.")
                    sys.exit(1)
            if direction == 'up':
                extreme_y = y_coords.min()
            else:
                extreme_y = y_coords.max()
            extreme_point = (centroid_x, extreme_y)
        else:
            print(f"Error: Unknown direction '{direction}'.")
            sys.exit(1)
        return extreme_point

    # Find extreme points in all four directions
    left_point = find_extreme_point(mask, centroid_x, centroid_y, 'left')
    right_point = find_extreme_point(mask, centroid_x, centroid_y, 'right')
    up_point = find_extreme_point(mask, centroid_x, centroid_y, 'up')
    down_point = find_extreme_point(mask, centroid_x, centroid_y, 'down')

    # Calculate radii in pixels
    radius_left_px = np.linalg.norm(np.array([left_point[0], left_point[1]]) - np.array([centroid_x, centroid_y]))
    radius_right_px = np.linalg.norm(np.array([right_point[0], right_point[1]]) - np.array([centroid_x, centroid_y]))
    radius_up_px = np.linalg.norm(np.array([up_point[0], up_point[1]]) - np.array([centroid_x, centroid_y]))
    radius_down_px = np.linalg.norm(np.array([down_point[0], down_point[1]]) - np.array([centroid_x, centroid_y]))

    radii_pixels = {
        'left': radius_left_px,
        'right': radius_right_px,
        'up': radius_up_px,
        'down': radius_down_px
    }

    # Extract focal length from camera matrix
    fx = camera_matrix[0, 0]
    if focal_length is None:
        focal_length_px = fx
    else:
        focal_length_px = focal_length

    # Calculate angular radii in radians
    angular_radii = {direction: radius / focal_length_px for direction, radius in radii_pixels.items()}

    # Convert angular radii to real-world radii
    radii_real = {direction: angular_radius * distance_to_planet for direction, angular_radius in angular_radii.items()}

    # Collect points
    points = {
        'left': left_point,
        'right': right_point,
        'up': up_point,
        'down': down_point
    }

    return radii_real, radii_pixels, points, (centroid_x, centroid_y)

def main():

    image_path = "./dataset/task4ab_image.jpg"
    mask_path =  "./dataset/task4ab_custom_mask.png"

    # image_path = r"C:\Users\ziyar\Computer Vision and Sensing\final project\pictures taken\six floor\image_3.jpg"             
    # mask_path = r"C:\Users\ziyar\Computer Vision and Sensing\final project\pictures taken\six floor_mask\image_3_mask.png"

    # For \pictures taken\images_side_view\image_70.jpg distance is 11

    distance_to_planet = 8.5 # Ensure the unit consistency (e.g., meters, kilometers)
    focal_length = (916.09 + 916.49) / 2  # Average focal length in pixels

    # Verify that the image and mask paths exist
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        sys.exit(1)
    if not os.path.exists(mask_path):
        print(f"Error: Mask file {mask_path} does not exist.")
        sys.exit(1)

    # Load camera parameters
    camera_matrix, dist_coeffs = load_camera_parameters()

    # Load the image (no undistortion)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        sys.exit(1)

    # Optionally, save or display the original image
    # cv2.imwrite("original_image.jpg", image)
    cv2.imshow("Original Image", image)
    cv2.waitKey(0)

    # Load the mask (assuming the mask corresponds to the original image)
    mask = load_mask(mask_path)

    # Optionally, display the mask
    mask_display = (mask * 255).astype(np.uint8)
    cv2.imshow("Mask", mask_display)
    cv2.waitKey(0)

    # Calculate the radii
    radii_real, radii_pixels, points, centroid = calculate_planet_radii(
        mask, camera_matrix, focal_length, distance_to_planet
    )

    print("Planet Radii:")
    for direction in ['left', 'right', 'up', 'down']:
        print(f"  {direction.capitalize()} Radius: {radii_real[direction]:.2f} units ({radii_pixels[direction]:.2f} px)")

    # Check consistency
    radii_values = list(radii_real.values())
    max_radius = max(radii_values)
    min_radius = min(radii_values)
    tolerance = 0.05  # 5% tolerance

    if (max_radius - min_radius) / max_radius <= tolerance:
        print("Radii are consistent within the tolerance.")
    else:
        print("Radii are NOT consistent within the tolerance.")

    # Optionally, visualize the radii on the image
    output_image = image.copy()

    # Draw centroid
    cv2.circle(output_image, (int(centroid[0]), int(centroid[1])), 5, (255, 0, 255), -1)  # Magenta

    # Draw radii lines
    colors = {
        'left': (255, 255, 0),   # Cyan
        'right': (255, 255, 0),  # Cyan
        'up': (255, 0, 255),     # Magenta
        'down': (255, 0, 255)    # Magenta
    }

    # Define font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8  # Increased font scale
    thickness = 2    # Increased thickness

    for direction, point in points.items():
        # Draw line
        cv2.line(output_image, (int(centroid[0]), int(centroid[1])), (int(point[0]), int(point[1])), colors[direction], 2)
        # Draw point
        cv2.circle(output_image, (int(point[0]), int(point[1])), 5, colors[direction], -1)
        # Prepare text with both real-world radius and pixel size
        # text = f"{direction.capitalize()} R: {radii_real[direction]:.2f} metres, {radii_pixels[direction]:.1f}px"#
        text = f"{direction.capitalize()} R: {radii_real[direction]:.2f} metres"
        # Calculate text size to adjust placement
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        # Set text position slightly offset from the point
        if direction in ['left', 'up']:
            text_pos = (int(point[0]) - text_width - 10, int(point[1]) - 10)
        else:
            text_pos = (int(point[0]) + 10, int(point[1]) + text_height + 10)
        # Put text on the image
        cv2.putText(output_image, text,
                    text_pos,
                    font, font_scale, colors[direction], thickness, cv2.LINE_AA)

    # Save or display the output image with radii
    cv2.imwrite("planet_with_radii.jpg", output_image)
    cv2.imshow("Planet Radii", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
        main()
