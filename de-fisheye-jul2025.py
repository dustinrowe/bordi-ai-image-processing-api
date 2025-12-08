import cv2
import numpy as np
import os

# Load image
image_path = '/Users/dflow/Downloads/test-de-fish.jpeg'
img = cv2.imread(image_path)

if img is None:
    print(f"Error: Could not load image from {image_path}")
    exit(1)

h, w = img.shape[:2]
print(f"Image loaded successfully. Dimensions: {w}x{h}")

# Camera matrix (guessing focal length)
K = np.array([
    [w * 3.0, 0, w/2],      # Increase f_x for more horizontal flattening
    [0, h * 5, h/2],
    [0, 0, 1]
])
D = np.array([
    -3,   # k1: positive to push edges out
    0.5,   # k2: small positive for edge fine-tuning
    0.5,     # p1: no tangential distortion
    0.1      # p2: no tangential distortion
])

# Remove distortion
map1, map2 = cv2.initUndistortRectifyMap(K, D, None, K, (w, h), cv2.CV_16SC2)
undistorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)

# Save result with descriptive filename
output_path = '/Users/dflow/Downloads/test-de-fish-undistorted.jpg'
cv2.imwrite(output_path, undistorted)
print(f"Undistorted image saved to: {output_path}")

# Optional: Display the images for comparison
cv2.imshow('Original', img)
cv2.imshow('Undistorted', undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
