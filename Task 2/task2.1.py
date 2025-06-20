import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
from task2_segment_funct import combined_segmentation_hough

# Load the image
image_path1 = './dataset/task2.1_front_view.jpg'
image1 = cv2.imread(image_path1)

image_path2 = './dataset/task2.1_right_view.jpg'
image2 = cv2.imread(image_path2)

# Find centroids
def find_points(final_mask):
    # Find contours in the binary mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        print("No contours found in the mask.")
        return None, None, None, None
    
    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Find the centers and the radius of the mask
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    
    # Convert circle parameters to integer values
    center_x = int(x)
    center_y = int(y)
    radius = int(radius)
    
    # Return the circle's center, radius, and the lowest point (x_min, y_min)
    return center_x, center_y, radius



earth_mask1, segmented_image1, image1  = combined_segmentation_hough(
        image_path = image_path1)

earth_mask2, segmented_image2, image2  = combined_segmentation_hough(
        image_path = image_path2)

earth_mask1 = (earth_mask1 > 0.001).astype(np.uint8) * 255
earth_mask2 = (earth_mask2 > 0.001).astype(np.uint8) * 255

center_x1, center_y1, radius1 = find_points(earth_mask1)
center_x2, center_y2, radius2 = find_points(earth_mask2)


# Draw on the image1
cv2.circle(image1, (center_x1, center_y1), radius1, (0, 255, 0), 2)
cv2.circle(image1, (center_x1, center_y1), 2, (0, 0, 255), 7)  
cv2.putText(image1, f" ({center_x1},{center_y1})", (center_x1, center_y1), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 255), 4)

# Draw on the image2
cv2.circle(image2, (center_x2, center_y2), radius2, (0, 255, 0), 2)
cv2.circle(image2, (center_x2, center_y2), 2, (0, 0, 255), 7)
cv2.putText(image2, f" ({center_x2},{center_y2})", (center_x2, center_y2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 255), 4)

# Draw on the mask1
earth_mask1 = np.zeros_like(image1, dtype=np.uint8)
cv2.circle(earth_mask1, (center_x1, center_y1), radius1, (255, 255, 255), -1)
cv2.circle(segmented_image1, (center_x1, center_y1), 4, (0, 0, 255), 7) 
cv2.putText(segmented_image1, f" ({center_x1},{center_y1})", (center_x1, center_y1), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 255), 5)
cv2.circle(earth_mask1, (center_x1, center_y1), 4, (0, 0, 255), 7) 
cv2.putText(earth_mask1, f" ({center_x1},{center_y1})", (center_x1, center_y1), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 255), 5)

# Draw on the mask2
earth_mask2 = np.zeros_like(image2, dtype=np.uint8)
cv2.circle(earth_mask2, (center_x2, center_y2), radius2, (255, 255, 255), -1)
cv2.circle(segmented_image2, (center_x2, center_y2), 4, (0, 0, 255), 7)
cv2.putText(segmented_image2, f" ({center_x2},{center_y2})", (center_x2, center_y2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 255), 5)
cv2.circle(earth_mask2, (center_x2, center_y2), 4, (0, 0, 255), 7)
cv2.putText(earth_mask2, f" ({center_x2},{center_y2})", (center_x2, center_y2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (30, 30, 255), 5)


# Converr to RGB
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
segmented_image1 = cv2.cvtColor(segmented_image1, cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
segmented_image2 = cv2.cvtColor(segmented_image2, cv2.COLOR_BGR2RGB)


# Show figures double
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image1)
ax[0].set_title("Countour Detection")
ax[1].imshow(segmented_image1)
ax[1].set_title("Segmentation on the Image")
plt.savefig('Results_Images/task2.1/countour_seg_front.png')
plt.show()  

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(image2)
ax[0].set_title("Countour Detection")
ax[1].imshow(segmented_image2)
ax[1].set_title("Segmentation on the Image")
plt.savefig('Results_Images/task2.1/countour_seg_side.png')
plt.show() 

print(f"Center of Earth in the image: ({center_x1}, {center_y1})")
print(f"Center of Earth in the image: ({center_x2}, {center_y2})")
