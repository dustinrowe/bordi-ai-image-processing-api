import cv2
import numpy as np

# Read the image
image = cv2.imread('/Users/dflow/Downloads/unnamed.jpg')
if image is None:
    print("Error: Could not load the image")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Since we know lines are black on white background, we can use thresholding
# Apply adaptive thresholding to separate black lines from white background
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operations to clean up the thresholded image
kernel = np.ones((2,2), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

# Apply edge detection on the cleaned binary image
edges = cv2.Canny(cleaned, 50, 150, apertureSize=3)

# Apply Hough transform with parameters optimized for black lines on white background
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=90, minLineLength=40, maxLineGap=20)

# Create a copy for drawing
result_image = image.copy()

# Draw all detected lines
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display results
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded (Black Lines)', thresh)
cv2.imshow('Cleaned Binary', cleaned)
cv2.imshow('Edge Detection', edges)
cv2.imshow('Grid Detection', result_image)

print(f"Total lines detected: {len(lines) if lines is not None else 0}")

cv2.waitKey(0)
cv2.destroyAllWindows()
