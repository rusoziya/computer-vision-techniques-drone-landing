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
        # Handle grayscale images by converting to BGR
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
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
    # If image is grayscale, convert to BGR
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
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

def gamma_correction(image, gamma):
    """
    Applies gamma correction to the input image.

    Parameters:
        image (numpy.ndarray): Input image.
        gamma (float): Gamma value for correction.

    Returns:
        numpy.ndarray: Gamma corrected image.
    """
    if gamma <= 0:
        return image
    inv_gamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** inv_gamma) * 255
                       for i in np.arange(256)]).astype("uint8"))
    return cv2.LUT(image, table)

def equalize_value_channel(hsv_img):
    """
    Equalizes the V channel of an HSV image.

    Parameters:
        hsv_img (numpy.ndarray): HSV image.

    Returns:
        numpy.ndarray: HSV image with equalized V channel.
    """
    h, s, v = cv2.split(hsv_img)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return hsv_eq

def draw_boundaries_and_ellipse(image, mask):
    """
    Draws red boundaries around contours and a green ellipse around the largest contour.

    Parameters:
        image (numpy.ndarray): Original image.
        mask (numpy.ndarray): Binary mask.

    Returns:
        numpy.ndarray: Image with drawn boundaries and ellipse.
    """
    output = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Draw all contours in red
        cv2.drawContours(output, contours, -1, (0, 0, 255), 2)

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        if len(largest_contour) >= 5:
            # Fit an ellipse to the largest contour
            ellipse = cv2.fitEllipse(largest_contour)
            cv2.ellipse(output, ellipse, (0, 255, 0), 2)  # Green ellipse

    return output

def interactive_hsv_segmentation(image_path, save_directory):
    """
    Opens an interactive window with sliders to adjust HSV segmentation parameters, gamma correction,
    histogram equalization, and morphological operations. Displays various processed images in real-time.

    Args:
        image_path (str): Path to the input image.
        save_directory (str): Directory where the final stacked image will be saved.
    """
    # Load image and check if successful
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Failed to load image at: {image_path}")
        return

    # Define default parameters
    # HSV range
    DEFAULT_L_H = 57
    DEFAULT_L_S = 15
    DEFAULT_L_V = 0
    DEFAULT_U_H = 110
    DEFAULT_U_S = 255
    DEFAULT_U_V = 255

    # Morphological operations
    DEFAULT_EROSION_ITER = 2
    DEFAULT_DILATION_ITER = 5
    KERNEL_SIZE = (13, 13)

    # Preprocessing
    DEFAULT_GAMMA = 1.15
    DEFAULT_HIST_EQUALIZE = True
    DEFAULT_BLUR_KERNEL_SIZE = 3

    # Create windows
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 600)

    # Create trackbars for HSV segmentation with default values
    cv2.createTrackbar("Lower H", "Controls", DEFAULT_L_H, 179, nothing)
    cv2.createTrackbar("Lower S", "Controls", DEFAULT_L_S, 255, nothing)
    cv2.createTrackbar("Lower V", "Controls", DEFAULT_L_V, 255, nothing)
    cv2.createTrackbar("Upper H", "Controls", DEFAULT_U_H, 179, nothing)
    cv2.createTrackbar("Upper S", "Controls", DEFAULT_U_S, 255, nothing)
    cv2.createTrackbar("Upper V", "Controls", DEFAULT_U_V, 255, nothing)

    # Create trackbar for gamma correction
    # Gamma range: 0.10 to 5.00, trackbar value: 10 to 500 (gamma = trackbar/100)
    cv2.createTrackbar("Gamma x100", "Controls", int(DEFAULT_GAMMA * 100), 500, nothing)

    # Create trackbar for histogram equalization toggle
    # 0 = Off, 1 = On
    cv2.createTrackbar("Hist Eq", "Controls", int(DEFAULT_HIST_EQUALIZE), 1, nothing)

    # Create trackbars for erosion and dilation iterations
    cv2.createTrackbar("Erosion Iter", "Controls", DEFAULT_EROSION_ITER, 10, nothing)
    cv2.createTrackbar("Dilation Iter", "Controls", DEFAULT_DILATION_ITER, 10, nothing)

    # Create a window for stacked images with resizable flag
    cv2.namedWindow("Stacked Images", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Stacked Images", 1920, 1080)  # Set to 1080p resolution

    while True:
        # Clone the original image for processing
        image = original_image.copy()

        # Get current positions of all trackbars
        l_h = cv2.getTrackbarPos("Lower H", "Controls")
        l_s = cv2.getTrackbarPos("Lower S", "Controls")
        l_v = cv2.getTrackbarPos("Lower V", "Controls")
        u_h = cv2.getTrackbarPos("Upper H", "Controls")
        u_s = cv2.getTrackbarPos("Upper S", "Controls")
        u_v = cv2.getTrackbarPos("Upper V", "Controls")
        gamma = max(cv2.getTrackbarPos("Gamma x100", "Controls") / 100.0, 0.10)  # Prevent gamma < 0.10
        hist_eq = cv2.getTrackbarPos("Hist Eq", "Controls")
        erosion_iter = cv2.getTrackbarPos("Erosion Iter", "Controls")
        dilation_iter = cv2.getTrackbarPos("Dilation Iter", "Controls")

        # Apply gamma correction
        gamma_corrected = gamma_correction(image, gamma)

        # Apply median blur
        if DEFAULT_BLUR_KERNEL_SIZE > 1 and DEFAULT_BLUR_KERNEL_SIZE % 2 == 1:
            blurred = cv2.medianBlur(gamma_corrected, DEFAULT_BLUR_KERNEL_SIZE)
        else:
            blurred = gamma_corrected.copy()

        # Convert to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Apply histogram equalization if enabled
        if hist_eq:
            hsv = equalize_value_channel(hsv)

        # Convert back to BGR for display
        if hist_eq:
            image_eq = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            image_eq = blurred.copy()

        # HSV segmentation
        lower_bound = (l_h, l_s, l_v)
        upper_bound = (u_h, u_s, u_v)
        mask_hsv = cv2.inRange(hsv, lower_bound, upper_bound)

        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE)
        mask_eroded = cv2.erode(mask_hsv, kernel, iterations=erosion_iter) if erosion_iter > 0 else mask_hsv.copy()
        mask_dilated = cv2.dilate(mask_eroded, kernel, iterations=dilation_iter) if dilation_iter > 0 else mask_eroded.copy()

        final_mask = mask_dilated.copy()

        # Draw boundaries and ellipse on original image
        image_with_boundaries = draw_boundaries_and_ellipse(original_image, final_mask)

        # Prepare images for stacking
        img_original_with_mask = add_title(image_with_boundaries, "Original with HSV Mask & Ellipse")
        img_gamma = add_title(gamma_corrected, "Gamma Corrected Image")
        img_blurred = add_title(blurred, "Median Blurred Image")
        img_hist_eq = add_title(image_eq, "Histogram Equalized Image")

        # Convert grayscale masks to BGR before adding titles
        mask_eroded_bgr = cv2.cvtColor(mask_eroded, cv2.COLOR_GRAY2BGR)
        mask_dilated_bgr = cv2.cvtColor(mask_dilated, cv2.COLOR_GRAY2BGR)

        # Add titles to the masks
        img_eroded = add_title(mask_eroded_bgr, "Eroded Mask")
        img_dilated = add_title(mask_dilated_bgr, "Dilated Mask")

        # Stack images in a 3x2 grid while preserving aspect ratios
        imgStack = stackImages(0.5, [
            [img_original_with_mask, img_gamma],
            [img_blurred, img_hist_eq],
            [img_eroded, img_dilated]
        ])

        # Display the stacked images
        cv2.imshow("Stacked Images", imgStack)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # Save the stacked image before exiting
            save_path = os.path.join(save_directory, "stacked_images.png")
            # Calculate the desired size based on 300 DPI and A4 dimensions (8.27 x 11.69 inches)
            # 300 DPI * 8.27 inches = 2481 pixels width
            # 300 DPI * 11.69 inches = 3507 pixels height
            desired_width = 2481
            desired_height = 3507
            resized_imgStack = cv2.resize(imgStack, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(save_path, resized_imgStack)
            print(f"Saved stacked image to: {save_path}")
            break
        # No other key functionality

    # Destroy all OpenCV windows
    cv2.destroyAllWindows()

def main():
    # Specify the path to your input image
   
    # image_path = r"C:\Users\ziyar\Computer Vision and Sensing\final project\task4\captured\captured_images_ao1\image_11.jpg"
    image_path = "./dataset/images/000003.png"
    # Specify the directory where you want to save the stacked image
    save_directory = "./task1_results/images"

    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    interactive_hsv_segmentation(image_path, save_directory)

if __name__ == "__main__":
    main()
