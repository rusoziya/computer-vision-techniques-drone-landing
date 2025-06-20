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

def calculate_planet_diameter_left_right(mask, camera_matrix, focal_length, distance_to_planet):
    """
    Calculate the planet's diameter by finding the furthest left and right points in the center of the binary mask.

    Parameters:
    - mask: Binary mask of the planet (1 for planet, 0 for background).
    - camera_matrix: Camera matrix containing focal lengths.
    - focal_length: Focal length in pixels.
    - distance_to_planet: Distance from the camera to the planet in the same unit as desired for diameter.

    Returns:
    - diameter_real: Diameter of the planet in real-world units.
    - diameter_pixels: Diameter of the planet in pixels.
    - left_point: Leftmost point defining the diameter.
    - right_point: Rightmost point defining the diameter.
    """
    # Extract all foreground pixels
    foreground_pixels = np.column_stack(np.where(mask == 1))
    if foreground_pixels.size == 0:
        print("Error: No foreground pixels found in the mask.")
        sys.exit(1)

    # Calculate centroid of the foreground
    centroid = foreground_pixels.mean(axis=0)
    centroid_y = int(round(centroid[0]))  # y-coordinate (row)

    # Handle edge cases where centroid_y might be out of bounds
    centroid_y = max(0, min(centroid_y, mask.shape[0] - 1))

    # Find all foreground pixels at the centroid row
    x_coords = np.where(mask[centroid_y, :] == 1)[0]

    if len(x_coords) == 0:
        # If no pixels at centroid_y, search nearby rows
        search_radius = 5  # Number of rows to search above and below
        found = False
        for offset in range(1, search_radius + 1):
            # Search above
            if centroid_y - offset >= 0:
                x_coords = np.where(mask[centroid_y - offset, :] == 1)[0]
                if len(x_coords) > 0:
                    centroid_y = centroid_y - offset
                    found = True
                    break
            # Search below
            if centroid_y + offset < mask.shape[0]:
                x_coords = np.where(mask[centroid_y + offset, :] == 1)[0]
                if len(x_coords) > 0:
                    centroid_y = centroid_y + offset
                    found = True
                    break
        if not found:
            print("Error: No foreground pixels found near the centroid row.")
            sys.exit(1)

    # Now, x_coords contains all x positions at the adjusted centroid_y
    left_x = x_coords.min()
    right_x = x_coords.max()
    left_point = (left_x, centroid_y)  # (x, y)
    right_point = (right_x, centroid_y)  # (x, y)

    # Calculate Euclidean distance between left and right points
    distance_pixels = np.linalg.norm(np.array(left_point) - np.array(right_point))
    diameter_pixels = distance_pixels

    # Extract focal length from camera matrix
    fx = camera_matrix[0, 0]
    if focal_length is None:
        focal_length_px = fx
    else:
        focal_length_px = focal_length

    # Calculate angular diameter in radians
    angular_diameter = diameter_pixels / focal_length_px  # radians

    # Convert angular diameter to real-world diameter
    diameter_real = angular_diameter * distance_to_planet  # same unit as distance

    return diameter_real, diameter_pixels, left_point, right_point

def calculate_planet_diameter(mask, camera_matrix, focal_length, distance_to_planet, method='left_right'):
    """
    Calculate the planet's diameter in real-world units using the specified method.

    Parameters:
    - mask: Binary mask of the planet (1 for planet, 0 for background).
    - camera_matrix: Camera matrix containing focal lengths.
    - focal_length: Focal length in pixels.
    - distance_to_planet: Distance from the camera to the planet in the same unit as desired for diameter.
    - method: Method to calculate diameter ('left_right').

    Returns:
    - diameter_real: Diameter of the planet in real-world units.
    - diameter_pixels: Diameter of the planet in pixels.
    - point1: First point defining the diameter.
    - point2: Second point defining the diameter.
    """
    if method == 'left_right':
        return calculate_planet_diameter_left_right(mask, camera_matrix, focal_length, distance_to_planet)
    else:
        print(f"Error: Unknown method '{method}'. Choose 'left_right'.")
        sys.exit(1)

def main():
    # Define your input parameters here
    # Example paths (ensure these paths are correct on your system)
    image_path = "./dataset/task4ab_image.jpg"
    mask_path =  "./dataset/task4ab_custom_mask.png"

    distance_to_planet = 8.5  # Ensure the unit consistency (e.g., meters, kilometers)
    focal_length = (916.09 + 916.49) / 2  # Average focal length in pixels

    # Choose the method for diameter calculation: currently only 'left_right' is supported
    method = 'left_right'  # Fixed to 'left_right' as per user request

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

    # Calculate the diameter using the selected method
    diameter_real, diameter_pixels, point1, point2 = calculate_planet_diameter(
        mask, camera_matrix, focal_length, distance_to_planet, method=method
    )

    print(f"Planet Diameter: {diameter_real:.2f} units (based on distance units)")
    print(f"Diameter in Image: {diameter_pixels:.2f} pixels")

    # Define font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85        # Increased font scale as per requirement
    thickness = 2          # Increased thickness as per requirement
    color_cyan = (0, 255, 255)  # Cyan color in BGR

    # Optionally, visualize the result
    # Draw the diameter line on the original image
    output_image = image.copy()

    # Ensure points are integers
    point1 = tuple(map(int, point1))
    point2 = tuple(map(int, point2))

    # Draw the diameter line with cyan color
    cv2.line(output_image, point1, point2, color_cyan, thickness)

    # Calculate midpoint for placing text
    midpoint = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)

    # Prepare the text to display
    text = f"Pixel Diameter: {diameter_pixels:.2f}px, Real Diameter: {diameter_real:.2f} metres"

    # Determine text size to adjust text placement if needed
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Adjust text position to ensure it doesn't go out of image bounds
    text_x = max(0, midpoint[0] - text_width // 2)
    text_y = max(text_height, midpoint[1] - 10)  # 10 pixels above the midpoint

    # Put the text on the image with cyan color
    cv2.putText(output_image, text, 
                (text_x, text_y),
                font, font_scale, color_cyan, thickness, cv2.LINE_AA)

    # Save or display the output image
    cv2.imwrite("planet_with_diameter.jpg", output_image)
    cv2.imshow("Planet Diameter", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
