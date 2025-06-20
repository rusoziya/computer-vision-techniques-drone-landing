import cv2
import numpy as np
import os

def stackImages(scale, imgArray):
    """
    Stacks images either horizontally or vertically for display while preserving aspect ratios.

    Parameters:
        scale (float): Scaling factor for resizing images.
        imgArray (list): List containing images to be stacked. Images can be arranged either horizontally or vertically.

    Returns:
        numpy.ndarray: Stacked image.
    """
    # Determine if imgArray is a 2D list (grid) or 1D list
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)

    # Calculate target width and height based on the first image after scaling
    if rowsAvailable:
        first_img = imgArray[0][0]
    else:
        first_img = imgArray[0]
    target_width = int(first_img.shape[1] * scale)
    target_height = int(first_img.shape[0] * scale)

    def resize_and_pad(img, target_width, target_height):
        """
        Resizes an image while maintaining aspect ratio and pads it to match target dimensions.

        Parameters:
            img (numpy.ndarray): Image to resize and pad.
            target_width (int): Target width after scaling.
            target_height (int): Target height after scaling.

        Returns:
            numpy.ndarray: Resized and padded image.
        """
        height, width = img.shape[:2]
        aspect_ratio = width / height

        # Determine new dimensions while maintaining aspect ratio
        if width > height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        # Resize the image
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create a black canvas
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

        # Compute top-left corner for centering the image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2

        # Place the resized image onto the canvas
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img

        return canvas

    if rowsAvailable:
        # Initialize list to hold horizontally stacked rows
        stacked_rows = []
        for row in imgArray:
            # Resize and pad each image in the row
            resized_row = [resize_and_pad(img, target_width, target_height) for img in row]
            # Horizontally stack images in the row
            hor = np.hstack(resized_row)
            stacked_rows.append(hor)
        # Vertically stack all rows
        ver = np.vstack(stacked_rows)
    else:
        # For a single row of images
        resized_imgs = [resize_and_pad(img, target_width, target_height) for img in imgArray]
        ver = np.hstack(resized_imgs)

    return ver

def nothing(x):
    pass  # Dummy function for trackbar callback

def add_title(image, title, position=(10, 50)):
    """
    Adds a title to the top of an image.

    Parameters:
        image (numpy.ndarray): The image to annotate.
        title (str): The text to display as the title.
        position (tuple): The (x, y) position for the text.

    Returns:
        numpy.ndarray: The annotated image.
    """
    annotated_image = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2  # Increased font scale for larger titles
    font_color = (0, 255, 0)  # Green color for visibility
    thickness = 3  # Increased thickness for better visibility
    line_type = cv2.LINE_AA

    # Calculate the size of the text box
    text_size, _ = cv2.getTextSize(title, font, font_scale, thickness)
    text_width, text_height = text_size

    # Add a filled rectangle as background for the text
    cv2.rectangle(
        annotated_image,
        (position[0] - 10, position[1] - text_height - 20),
        (position[0] + text_width + 10, position[1] + 10),
        (0, 0, 0),  # Black background
        cv2.FILLED
    )

    # Put the text on top of the rectangle
    cv2.putText(
        annotated_image,
        title,
        position,
        font,
        font_scale,
        font_color,
        thickness,
        line_type
    )

    return annotated_image

def interactive_circle_detection(image_path, params, save_directory):
    """
    Opens an interactive window with sliders to adjust parameters for Hough Circle Transform.
    Performs circle detection and creates a binary mask based on detected circles.

    Args:
        image_path (str): Path to the input image.
        params (dict): Dictionary containing parameters for Gaussian Blur.
        save_directory (str): Directory where the final stacked image will be saved.
    """
    # Load image and check if successful
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image at: {image_path}")
        return

    # Step 1: Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, params['gaussian_kernel_size'], params['gaussian_sigma'])

    # The processed image to be used for Hough Circle Transform
    processed_img = blurred

    # Define Titles (4 images for a 2x2 grid)
    titles = [
        "Original Image with Detected Circles",
        "Blurred Image",
        "Canny Edges Detected",
        "Mask",
    ]

    # Create windows
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 300)  # Adjusted size for fewer trackbars

    # Create trackbars for Hough Circle Transform
    cv2.createTrackbar("Param1", "Controls", 100, 500, nothing)    # Higher threshold for internal Canny
    cv2.createTrackbar("Param2", "Controls", 30, 100, nothing)     # Accumulator threshold
    cv2.createTrackbar("DP", "Controls", 10, 30, nothing)          # dp = 1.0 to 3.0 (scaled by 10)
    cv2.createTrackbar("MinDist", "Controls", 20, 1000, nothing)   # Minimum distance between centers
    cv2.createTrackbar("MinRadius", "Controls", 0, 600, nothing)   # Minimum circle radius
    cv2.createTrackbar("MaxRadius", "Controls", 0, 1000, nothing)  # Maximum circle radius (0 = no limit)

    # Create a window for stacked images with resizable flag
    cv2.namedWindow("Stacked Images", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stacked Images", 1920, 1080)  # Set to 1080p resolution

    while True:
        # Get current positions of Hough Circle Transform trackbars
        hough_param1 = cv2.getTrackbarPos("Param1", "Controls")
        hough_param2 = cv2.getTrackbarPos("Param2", "Controls")
        hough_dp = cv2.getTrackbarPos("DP", "Controls")
        hough_minDist = cv2.getTrackbarPos("MinDist", "Controls")
        hough_minRadius = cv2.getTrackbarPos("MinRadius", "Controls")
        hough_maxRadius = cv2.getTrackbarPos("MaxRadius", "Controls")

        # Adjust dp parameter (scale by 10 to allow decimal values)
        dp = 1.0 + hough_dp / 10.0  # dp must be >0

        # Apply Hough Circle Transform on the processed image
        circles = cv2.HoughCircles(
            processed_img,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=hough_minDist,
            param1=hough_param1,   # Used internally by HoughCircles
            param2=hough_param2,   # Accumulator threshold for circle detection
            minRadius=hough_minRadius,
            maxRadius=hough_maxRadius if hough_maxRadius > 0 else 0
        )

        # Initialize mask
        mask = np.zeros_like(gray)

        # Draw detected circles on mask and original image
        display_image = image.copy()
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Draw filled circle on mask
                cv2.circle(mask, (i[0], i[1]), i[2], 255, -1)
                # Draw center and perimeter on display image
                cv2.circle(display_image, (i[0], i[1]), 2, (0, 255, 0), 3)    # Center
                cv2.circle(display_image, (i[0], i[1]), i[2], (0, 0, 255), 2) # Perimeter

        # Apply Canny Edge Detection based on Hough Param1
        canny_lower = int(hough_param1 / 2)
        canny_upper = hough_param1
        edges = cv2.Canny(processed_img, canny_lower, canny_upper)

        # Prepare images for stacking
        img_original = add_title(display_image, titles[0])
        img_blurred = add_title(cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR), titles[1])
        img_edges = add_title(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), titles[2])
        img_mask = add_title(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), titles[3])

        # Stack images in a 2x2 grid while preserving aspect ratios
        imgStack = stackImages(0.5, [
            [img_original, img_blurred],
            [img_edges, img_mask]
        ])

        # Display the stacked images
        cv2.imshow("Stacked Images", imgStack)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # Ensure the save directory exists
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            
            # Define the save path as "300.png"
            save_path = os.path.join(save_directory, "300.png")
            
            # Save the stacked image
            cv2.imwrite(save_path, imgStack)
            print(f"Stacked image saved at: {save_path}")
            break

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

# Example Usage
if __name__ == "__main__":
    # Define parameters
    params = {
        # Gaussian Blur Parameters
        'gaussian_kernel_size': (5, 5),  # Must be odd and positive
        'gaussian_sigma': 1.5,           # Standard deviation in X and Y direction
    }

    # Replace the image path with your own image
    image_path = "./dataset/images/000003.png"
    # Specify the directory where you want to save the stacked image
    save_directory = "./task1_results/images"

    interactive_circle_detection(image_path, params, save_directory)
