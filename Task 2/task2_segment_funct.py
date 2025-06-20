import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def gamma_correction(image, gamma):
    if gamma <= 0:
        return image
    inv_gamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** inv_gamma) * 255
                       for i in np.arange(256)]).astype("uint8"))
    return cv2.LUT(image, table)

def equalize_value_channel(hsv_img):
    h, s, v = cv2.split(hsv_img)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return hsv_eq


def combined_segmentation_hough(
    image_path,
    hough_param1 = 225,
    hough_param2 =46,
    dp =1.0,
    minDist = 1500,
    minRadius = 100,
    maxRadius=1000,
    #G_BLUR_KERNEL_SIZE=(3, 3),
    G_BLUR_KERNEL_SIZE=(5, 5),
    SIGMA=1.5,
):
    """
    Generate a Hough mask using Hough Circle detection.
 
    Returns:
    --------
    hough_mask : np.ndarray
        Binary mask from Hough Circle detection.
    segmented_image : np.ndarray
        Image with Hough mask applied to the original image.
    original_image : np.ndarray
        The original input image.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None, None
 
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, G_BLUR_KERNEL_SIZE, SIGMA)
 
    # Hough Circle Detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=hough_param1,
        param2=hough_param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
 
    # Initialize an empty Hough mask
    mask_hough = np.zeros_like(gray, dtype=np.float32)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        height, width = gray.shape
        Y, X = np.ogrid[:height, :width]
        for (cx, cy, r) in circles[0, :]:
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            # Create a circular mask with a gradient
            new_mask = np.clip(1 - (dist / r), 0, 1)
            # Maximize to handle multiple circles if present
            mask_hough = np.maximum(mask_hough, new_mask)
 
    # Binarize the mask
    hough_mask = (mask_hough > 0.000001).astype(np.uint8)
 
    # Create segmented image
    segmented_image = (hough_mask[..., None] * image).astype(np.uint8)
 
    return hough_mask, segmented_image, image

# New parameters (hough_param1, hough_param2)
def combined_segmentation_hough2(
    image_path,
    hough_param1 = 150,
    hough_param2 =30,
    dp =1.0,
    minDist = 1500,
    minRadius = 100,
    maxRadius=1000,
    #G_BLUR_KERNEL_SIZE=(3, 3),
    G_BLUR_KERNEL_SIZE=(5, 5),
    SIGMA=1.5,
):
    """
    Generate a Hough mask using Hough Circle detection.
 
    Returns:
    --------
    hough_mask : np.ndarray
        Binary mask from Hough Circle detection.
    segmented_image : np.ndarray
        Image with Hough mask applied to the original image.
    original_image : np.ndarray
        The original input image.
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None, None, None
 
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, G_BLUR_KERNEL_SIZE, SIGMA)
 
    # Hough Circle Detection
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=minDist,
        param1=hough_param1,
        param2=hough_param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )
 
    # Initialize an empty Hough mask
    mask_hough = np.zeros_like(gray, dtype=np.float32)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        height, width = gray.shape
        Y, X = np.ogrid[:height, :width]
        for (cx, cy, r) in circles[0, :]:
            dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
            # Create a circular mask with a gradient
            new_mask = np.clip(1 - (dist / r), 0, 1)
            # Maximize to handle multiple circles if present
            mask_hough = np.maximum(mask_hough, new_mask)
 
    # Binarize the mask
    hough_mask = (mask_hough > 0.000001).astype(np.uint8)
 
    # Create segmented image
    segmented_image = (hough_mask[..., None] * image).astype(np.uint8)
 
    return hough_mask, segmented_image, image