import cv2
import numpy as np
from matplotlib import pyplot as plt
from task2_segment_funct import combined_segmentation_hough2

# Read the images
im1_path = "./dataset/task2.3_image1_height.jpg"
im2_path = "./dataset/task2.3_image2_height.jpg"


im1_ = cv2.imread(im1_path)
im2 = cv2.imread(im2_path)

# Ensure images are the same size
im1 = cv2.resize(im1_, (im2.shape[1], im2.shape[0]))

# Apply combined segmentation
mask1, segmented_mask1, image1 = combined_segmentation_hough2(im1_path)
mask2, segmented_mask2, image2 = combined_segmentation_hough2(im2_path)


# Convert the masks to 8-bit
mask1 = (mask1 * 255).astype('uint8')
mask2 = (mask2 * 255).astype('uint8')
mask1 = cv2.convertScaleAbs(mask1)
mask2 = cv2.convertScaleAbs(mask2)

# Find the centroids
contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the centroids
contours1 = max(contours1, key=cv2.contourArea)
contours2 = max(contours2, key=cv2.contourArea)
(x1, y1), radius1 = cv2.minEnclosingCircle(contours1)
(x2, y2), radius2 = cv2.minEnclosingCircle(contours2)

x1, y1 = int(x1), int(y1)
x2, y2 = int(x2), int(y2)

# Display the centroids
image1_centroid = cv2.circle(segmented_mask1, (x1, y1), 10, (0, 0, 255), -1)
image2_centroid = cv2.circle(segmented_mask2, (x2, y2), 10, (0, 255, 0), -1)
# Write C1 and C2 on the centroids
cv2.putText(image1_centroid, 'C1', (x1-15, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(image2_centroid, 'C2', (x2-15, y2-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


plt.imshow(cv2.cvtColor(image1_centroid, cv2.COLOR_BGR2RGB)), plt.title("Image 1")
plt.savefig('Results_Images/task2.3/image1_centr.png')
plt.show()
plt.imshow(cv2.cvtColor(image2_centroid, cv2.COLOR_BGR2RGB)), plt.title("Image 2")
plt.savefig('Results_Images/task2.3/image2_centr.png')
plt.show()


# Display first image with both centroids (and line connecting them)
image_centroid = cv2.circle(image1, (x1, y1), 10, (0, 0, 255), -1)
cv2.putText(image_centroid, 'C1', (x1-15, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
image_centroid = cv2.circle(image1, (x2, y2), 10, (0, 255, 0), -1)
cv2.putText(image_centroid, 'C2', (x2-15, y2-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
image_centroid = cv2.line(image1, (x1, y1), (x2, y2), (10, 10, 10), 2)

plt.figure(figsize=(15, 15))
plt.imshow(cv2.cvtColor(image_centroid, cv2.COLOR_BGR2RGB)), plt.title("Image 1 with centroids")
plt.savefig('Results_Images/task2.3/image_centr_line.png')
plt.show()



# Compute the distance between centroids
distance_centroids = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Constants
BASELINE = 1           
FOCAL_LENGTH = 916.1

# Compute depth map from disparity
height = (FOCAL_LENGTH * BASELINE) / distance_centroids

print(f"Distance between centroids: {round(distance_centroids,3)} pixels")
print(f"Estimated Height: {round(height,3)} m")

