import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, jaccard_score

def gamma_correction(image, gamma):
    """Apply gamma correction to the input image.

    Parameters:
    -----------
    image : np.ndarray
        Input image (BGR format).
    gamma : float
        Gamma value for correction. Should be > 0.

    Returns:
    --------
    corrected_image : np.ndarray
        Gamma-corrected image.
    """
    if gamma <= 0:
        return image
    inv_gamma = 1.0 / gamma
    table = (np.array([((i / 255.0) ** inv_gamma) * 255
                       for i in np.arange(256)]).astype("uint8"))
    return cv2.LUT(image, table)

def equalize_value_channel(hsv_img):
    """Equalize the value (V) channel of an HSV image to improve contrast.

    Parameters:
    -----------
    hsv_img : np.ndarray
        Input HSV image.

    Returns:
    --------
    hsv_eq : np.ndarray
        HSV image with equalized V channel.
    """
    h, s, v = cv2.split(hsv_img)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge([h, s, v_eq])
    return hsv_eq

def combined_segmentation(
    image_path,
    hough_param1,
    hough_param2,
    dp,
    minDist,
    minRadius,
    maxRadius=1000,
    G_BLUR_KERNEL_SIZE=(5,5),
    SIGMA=1.5,
    l_h=57,
    l_s=15,
    l_v=33,
    u_h=110,
    u_s=255,
    u_v=240,
    erosion_iter=2,
    dilation_iter=5,
    kernel_size=(3,3),
    gamma=1.0,
    hist_equalize=False,
    blur_kernel_size=5
):
    """Perform combined segmentation using HSV thresholding and Hough Circle detection.

    Returns:
    --------
    dict
        {
            'final_mask': final binary mask after HSV+Hough combination,
            'original_image': the original preprocessed image,
            'mask_hough': the float mask from Hough detection,
            'mask_hsv': the float mask from HSV threshold
        }
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return None

    image = gamma_correction(image, gamma)
    if blur_kernel_size > 1 and blur_kernel_size % 2 == 1:
        image = cv2.medianBlur(image, blur_kernel_size)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if hist_equalize:
        hsv = equalize_value_channel(hsv)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, G_BLUR_KERNEL_SIZE, SIGMA)

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

    mask_hough = np.zeros_like(gray, dtype=np.float32)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        height, width = gray.shape
        Y, X = np.ogrid[:height, :width]
        for (cx, cy, r) in circles[0, :]:
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            new_mask = np.clip(1 - (dist / r), 0, 1)
            mask_hough = np.maximum(mask_hough, new_mask)

    lower_bound = (l_h, l_s, l_v)
    upper_bound = (u_h, u_s, u_v)
    mask_hsv = cv2.inRange(hsv, lower_bound, upper_bound).astype(np.float32) / 255.0

    combined_mask_continuous = mask_hough * mask_hsv
    # Combine the masks and normalize between 0 and 1

    combined_mask_continuous = combined_mask_continuous / combined_mask_continuous.max()  # Normalizing to [0, 1]

    # Convert to 8-bit representation for further processing
    combined_mask_8 = (combined_mask_continuous * 255).astype(np.uint8)

    # Perform morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    if erosion_iter > 0:
        combined_mask_8 = cv2.erode(combined_mask_8, kernel, iterations=erosion_iter)
    if dilation_iter > 0:
        combined_mask_8 = cv2.dilate(combined_mask_8, kernel, iterations=dilation_iter)

    # Threshold to create the final binary mask
    final_mask = (combined_mask_8 > (0 * 255)).astype(np.uint8)
    
    return {
        'final_mask': final_mask,
        'original_image': image,
        'mask_hough': mask_hough,
        'mask_hsv': mask_hsv
    }

def draw_red_boundary(image, mask):
    """Draw a red boundary around the specified mask on the given image."""
    output = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(output, contours, -1, (0,0,255), 2)
    return output

def main():
    # TASK1 Directories
    # Put the Segmentation Test set here
    input_images_dir = "./dataset/images"
                         
    # Combined (HSV+Hough)
    hsv_hough_masks_dir = "./task1_results/hsv_hough_masks"
    hsv_hough_images_dir = "./task1_results/hsv_hough_images"

    # HSV & Hough & Contour & Ellipse
    hsv_hough_contour_ellipse_masks_dir = "./task1_results/hsv_hough_contour_ellipse_masks"
    hsv_hough_contour_ellipse_images_dir = "./task1_results/hsv_hough_contour_ellipse_images"

    # Individual directories
    hsv_masks_dir = "./task1_results/hsv_masks"
    hsv_images_dir = "./task1_results/hsv_images"
    hough_masks_dir = "./task1_results/hough_masks"
    hough_overlay_dir = "./task1_results/hough_images"

    # HSV contour + ellipse directories (NO HOUGH)
    hsv_contour_ellipse_masks_dir = "./task1_results/hsv_contour_ellipse_masks"
    hsv_contour_ellipse_images_dir = "./task1_results/hsv_contour_ellipse_images"

    # # TASK4 Directories
    # input_images_dir = "./task4/captured/images/ao1"

    # # Combined (HSV+Hough)
    # hsv_hough_masks_dir = "./task4/hsv_hough_masks"
    # hsv_hough_images_dir = "./task4/hsv_hough_images"

    # # HSV & Hough & Contour & Ellipse
    # hsv_hough_contour_ellipse_masks_dir = "./task4/hsv_hough_contour_ellipse_masks"
    # hsv_hough_contour_ellipse_images_dir = "./task4/hsv_hough_contour_ellipse_images"

    # # Individual directories
    # hsv_masks_dir = "./task4/hsv_masks"
    # hsv_images_dir = "./task4/hsv_images"
    # hough_masks_dir = "./task4/hough_masks"
    # hough_overlay_dir = "./task4/hough_images"

    # # HSV contour + ellipse directories (NO HOUGH)
    # hsv_contour_ellipse_masks_dir = "./task4/hsv_contour_ellipse_masks"
    # hsv_contour_ellipse_images_dir = "./task4/hsv_contour_ellipse_images"

    # Create directories if they don't exist
    dirs_to_create = [
        hsv_hough_masks_dir, hsv_hough_images_dir,
        hsv_hough_contour_ellipse_masks_dir, hsv_hough_contour_ellipse_images_dir,
        hsv_masks_dir, hough_masks_dir, hough_overlay_dir,
        hsv_contour_ellipse_masks_dir, hsv_contour_ellipse_images_dir,
        hsv_images_dir
    ]
    for d in dirs_to_create:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory at {d}")

    # Parameters
    HOUGH_PARAM1 = 225
    HOUGH_PARAM2 = 46
    DP = 1.0
    MINDIST = 1500
    MINRADIUS = 100
    MAXRADIUS = 1500
    G_BLUR_KERNEL_SIZE = (5,5)
    SIGMA = 1.5

    # # HSV range
    L_H = 57
    L_S = 15
    L_V = 10
    U_H = 110
    U_S = 255
    U_V = 255

    # # HSV range - Used for Task 4
    # L_H = 30
    # L_S = 0
    # L_V = 88
    # U_H = 121
    # U_S = 255
    # U_V = 255

    # Morph ops
    EROSION_ITER = 2
    DILATION_ITER = 5
    KERNEL_SIZE = (13, 13)

    # Preprocessing
    GAMMA = 1.15
    HIST_EQUALIZE = True
    BLUR_KERNEL_SIZE = 3

    supported_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    for filename in os.listdir(input_images_dir):
        if not filename.lower().endswith(supported_ext):
            print(f"Skipping unsupported file: {filename}")
            continue

        image_path = os.path.join(input_images_dir, filename)
        
        # Combined segmentation (HSV+Hough)
        result = combined_segmentation(
            image_path=image_path,
            hough_param1=HOUGH_PARAM1,
            hough_param2=HOUGH_PARAM2,
            dp=DP,
            minDist=MINDIST,
            minRadius=MINRADIUS,
            maxRadius=MAXRADIUS,
            G_BLUR_KERNEL_SIZE=G_BLUR_KERNEL_SIZE,
            SIGMA=SIGMA,
            l_h=L_H,
            l_s=L_S,
            l_v=L_V,
            u_h=U_H,
            u_s=U_S,
            u_v=U_V,
            erosion_iter=EROSION_ITER,
            dilation_iter=DILATION_ITER,
            kernel_size=KERNEL_SIZE,
            gamma=GAMMA,
            hist_equalize=HIST_EQUALIZE,
            blur_kernel_size=BLUR_KERNEL_SIZE
        )

        if result is None:
            print(f"Segmentation failed for {filename}, skipping.")
            continue

        # final_mask (HSV+Hough)
        final_mask = result['final_mask']
        original_image = result['original_image'].copy()
        h, w = original_image.shape[:2]

        # Save hsv_hough mask
        mask_filename = os.path.splitext(filename)[0] + '_mask.png'
        mask_path = os.path.join(hsv_hough_masks_dir, mask_filename)
        mask_to_save = (final_mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_to_save)
        print(f"Saved hsv_hough mask for {filename} to {mask_path}")

        # hsv_hough boundary image
        hsv_hough_boundary = draw_red_boundary(original_image, final_mask)
        hsv_hough_overlay_filename = os.path.splitext(filename)[0] + '_boundary.png'
        hsv_hough_overlay_path = os.path.join(hsv_hough_images_dir, hsv_hough_overlay_filename)
        cv2.imwrite(hsv_hough_overlay_path, hsv_hough_boundary)
        print(f"Saved hsv_hough boundary image for {filename} to {hsv_hough_overlay_path}")

        # hough mask
        hough_mask = (result['mask_hough'] > 0.000001).astype(np.uint8) * 255
        hough_mask_filename = os.path.splitext(filename)[0] + '_hough_mask.png'
        hough_mask_path = os.path.join(hough_masks_dir, hough_mask_filename)
        cv2.imwrite(hough_mask_path, hough_mask)
        print(f"Saved hough mask for {filename} to {hough_mask_path}")

        # hsv mask
        hsv_mask = (result['mask_hsv'] * 255).astype(np.uint8)
        hsv_mask_filename = os.path.splitext(filename)[0] + '_hsv_mask.png'
        hsv_mask_path = os.path.join(hsv_masks_dir, hsv_mask_filename)
        cv2.imwrite(hsv_mask_path, hsv_mask)
        print(f"Saved hsv mask for {filename} to {hsv_mask_path}")

        # hsv image (boundary)
        hsv_boundary = draw_red_boundary(original_image, hsv_mask)
        hsv_boundary_filename = os.path.splitext(filename)[0] + '_hsv_boundary.png'
        hsv_boundary_path = os.path.join(hsv_images_dir, hsv_boundary_filename)
        cv2.imwrite(hsv_boundary_path, hsv_boundary)
        print(f"Saved hsv boundary image for {filename} to {hsv_boundary_path}")

        # hough boundary
        hough_binary_mask = (hough_mask > 0).astype(np.uint8)
        hough_boundary = draw_red_boundary(original_image, hough_binary_mask)
        hough_overlay_filename = os.path.splitext(filename)[0] + '_hough_boundary.png'
        hough_overlay_path = os.path.join(hough_overlay_dir, hough_overlay_filename)
        cv2.imwrite(hough_overlay_path, hough_boundary)
        print(f"Saved hough boundary image for {filename} to {hough_overlay_path}")

        # ------------------------------------------------------
        # HSV+Hough+Contour+Ellipse (final_mask used here)
        # ------------------------------------------------------
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # hsv_hough_contour_ellipse mask
            hsv_hough_contour_ellipse_mask = np.zeros((h, w), dtype=np.uint8)
            # Start with ellipse mask as just the contour initially
            cv2.drawContours(hsv_hough_contour_ellipse_mask, [largest_contour], -1, 255, -1)

            hsv_hough_contour_ellipse_mask_filename = os.path.splitext(filename)[0] + '_hsv_hough_contour_ellipse_mask.png'
            hsv_hough_contour_ellipse_mask_path = os.path.join(hsv_hough_contour_ellipse_masks_dir, hsv_hough_contour_ellipse_mask_filename)
            cv2.imwrite(hsv_hough_contour_ellipse_mask_path, hsv_hough_contour_ellipse_mask)
            print(f"Saved hsv_hough_contour_ellipse mask for {filename} to {hsv_hough_contour_ellipse_mask_path}")

            hsv_hough_contour_ellipse_image = original_image.copy()
            cv2.drawContours(hsv_hough_contour_ellipse_image, [largest_contour], -1, (0,0,255), 2)
            hsv_hough_contour_ellipse_image_filename = os.path.splitext(filename)[0] + '_hsv_hough_contour_ellipse_boundary.png'
            hsv_hough_contour_ellipse_image_path = os.path.join(hsv_hough_contour_ellipse_images_dir, hsv_hough_contour_ellipse_image_filename)
            cv2.imwrite(hsv_hough_contour_ellipse_image_path, hsv_hough_contour_ellipse_image)
            print(f"Saved hsv_hough_contour_ellipse boundary image for {filename} to {hsv_hough_contour_ellipse_image_path}")

            if len(largest_contour) >= 15:
                ellipse = cv2.fitEllipse(largest_contour)

                # Overwrite with ellipse-only mask
                hsv_hough_contour_ellipse_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(hsv_hough_contour_ellipse_mask, ellipse, 255, -1)
                cv2.imwrite(hsv_hough_contour_ellipse_mask_path, hsv_hough_contour_ellipse_mask)
                print(f"Saved hsv_hough_contour_ellipse mask with ellipse for {filename} to {hsv_hough_contour_ellipse_mask_path}")

                cv2.ellipse(hsv_hough_contour_ellipse_image, ellipse, (0,0,255), 2)
                cv2.drawContours(hsv_hough_contour_ellipse_image, [largest_contour], -1, (0,255,0), 2)
                cv2.imwrite(hsv_hough_contour_ellipse_image_path, hsv_hough_contour_ellipse_image)
                print(f"Saved hsv_hough_contour_ellipse boundary image with ellipse for {filename} to {hsv_hough_contour_ellipse_image_path}")
            else:
                print(f"Not enough points to fit an ellipse for {filename}")
        else:
            print(f"No contours found to fit an ellipse for {filename}")

        # ------------------------------------------------------
        # HSV+Contour+Ellipse (NO Hough)
        # For this, we use only hsv_mask and create a final hsv-only mask.
        # Apply same morphological ops to hsv_mask to get hsv_only_final_mask.
        hsv_mask_8 = hsv_mask.copy()  # hsv_mask already uint8
        # Morph ops on hsv-only mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, KERNEL_SIZE)
        if EROSION_ITER > 0:
            hsv_mask_8 = cv2.erode(hsv_mask_8, kernel, iterations=EROSION_ITER)
        if DILATION_ITER > 0:
            hsv_mask_8 = cv2.dilate(hsv_mask_8, kernel, iterations=DILATION_ITER)

        hsv_only_final_mask = (hsv_mask_8 > 0).astype(np.uint8)

        # Contour and ellipse on hsv_only_final_mask
        contours_hsv_only, _ = cv2.findContours(hsv_only_final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_hsv_only:
            largest_contour_hsv_only = max(contours_hsv_only, key=cv2.contourArea)

            # hsv_contour_ellipse mask
            hsv_contour_ellipse_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(hsv_contour_ellipse_mask, [largest_contour_hsv_only], -1, 255, -1)

            hsv_contour_ellipse_mask_filename = os.path.splitext(filename)[0] + '_hsv_contour_ellipse_mask.png'
            hsv_contour_ellipse_mask_path = os.path.join(hsv_contour_ellipse_masks_dir, hsv_contour_ellipse_mask_filename)
            cv2.imwrite(hsv_contour_ellipse_mask_path, hsv_contour_ellipse_mask)
            print(f"Saved hsv_contour_ellipse mask for {filename} to {hsv_contour_ellipse_mask_path}")

            hsv_contour_ellipse_image = original_image.copy()
            cv2.drawContours(hsv_contour_ellipse_image, [largest_contour_hsv_only], -1, (0,0,255), 2)
            hsv_contour_ellipse_image_filename = os.path.splitext(filename)[0] + '_hsv_contour_ellipse_boundary.png'
            hsv_contour_ellipse_image_path = os.path.join(hsv_contour_ellipse_images_dir, hsv_contour_ellipse_image_filename)
            cv2.imwrite(hsv_contour_ellipse_image_path, hsv_contour_ellipse_image)
            print(f"Saved hsv_contour_ellipse boundary image for {filename} to {hsv_contour_ellipse_image_path}")

            if len(largest_contour_hsv_only) >= 5:
                ellipse_hsv_only = cv2.fitEllipse(largest_contour_hsv_only)

                # ellipse-only mask for hsv_contour_ellipse
                hsv_contour_ellipse_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(hsv_contour_ellipse_mask, ellipse_hsv_only, 255, -1)
                cv2.imwrite(hsv_contour_ellipse_mask_path, hsv_contour_ellipse_mask)
                print(f"Saved hsv_contour_ellipse mask with ellipse for {filename} to {hsv_contour_ellipse_mask_path}")

                cv2.ellipse(hsv_contour_ellipse_image, ellipse_hsv_only, (0,0,255), 2)
                cv2.drawContours(hsv_contour_ellipse_image, [largest_contour_hsv_only], -1, (0,255,0), 2)
                cv2.imwrite(hsv_contour_ellipse_image_path, hsv_contour_ellipse_image)
                print(f"Saved hsv_contour_ellipse boundary image with ellipse for {filename} to {hsv_contour_ellipse_image_path}")
            else:
                print(f"Not enough points to fit an ellipse (HSV-only) for {filename}")
        else:
            print(f"No contours found to fit an ellipse (HSV-only) for {filename}")

if __name__ == "__main__":
    main()
