"""
Day 12 - Computer Vision & Digital Image Processing (OpenCV)
=============================================================
Scenarios:
  1. Drone Camera Image Analysis (image properties + BGR->RGB + edges)
  2. Scanned Document Processing (denoise + threshold + contours)
  3. Smart Dashcam Lane Detection (Canny + Hough Transform)

Note: Real image files are not available locally, so numpy arrays
are used to simulate realistic images for each scenario.
"""

import numpy as np
import cv2

# ─────────────────────────────────────────────────────────────
# SCENARIO 1: Drone Camera Image Analysis
# ─────────────────────────────────────────────────────────────
# You received a JPG photo from a drone camera.
# Before running any computer vision pipeline on it, you need to:
#   1. Understand the image dimensions and color space
#   2. Check the pixel data type (uint8 = 0 to 255 per channel)
#   3. Convert BGR (OpenCV default) to RGB (for matplotlib display)
#   4. Convert to grayscale for further processing
#   5. Apply Gaussian Blur to reduce noise
#   6. Run Canny Edge Detection to find object boundaries
# This is the standard first step in any drone image analysis pipeline.
print("=" * 60)
print("SCENARIO 1: Drone Camera Image Analysis")
print("Task: Inspect image properties, convert color spaces, detect edges")
print("=" * 60)

# Simulating a drone photo (757x1162 BGR image)
np.random.seed(42)
img = np.random.randint(0, 256, (757, 1162, 3), dtype=np.uint8)

print(f"Shape : {img.shape}")    # (height, width, channels)
print(f"Dtype : {img.dtype}")    # uint8 -> values 0-255
print(f"Size  : {img.size}")     # total pixel count

# BGR -> RGB conversion
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(f"RGB Shape    : {img_rgb.shape}")

# BGR -> Grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"Grayscale Shape : {img_gray.shape}")

# Basic pixel statistics
print(f"Mean pixel value : {img.mean():.2f}")
print(f"Max pixel value  : {img.max()}")
print(f"Min pixel value  : {img.min()}")

# Gaussian Blur (reduces noise before edge detection)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
print(f"Blurred image shape : {img_blur.shape}")

# Canny Edge Detection
edges = cv2.Canny(img_gray, 100, 200)
print(f"Edge map shape              : {edges.shape}")
print(f"Non-zero edge pixels found  : {np.count_nonzero(edges)}")

print("\nDrone image analysis complete!")

# ─────────────────────────────────────────────────────────────
# SCENARIO 2: Scanned Document Processing
# ─────────────────────────────────────────────────────────────
# A company scans physical documents and wants to digitize them.
# Scanned images often have noise, uneven lighting, and gray areas.
# The processing pipeline:
#   1. Denoise with Gaussian Blur
#   2. Binarize with simple thresholding (black text, white background)
#   3. Adaptive thresholding for uneven lighting conditions
#   4. Morphological opening to remove small noise dots
#   5. Find contours to detect and count text regions
# This is used in OCR (Optical Character Recognition) preprocessing.
print("\n" + "=" * 60)
print("SCENARIO 2: Scanned Document Processing")
print("Task: Denoise -> Threshold -> Clean -> Detect text regions")
print("=" * 60)

# Simulating a scanned document (white background, dark text lines)
np.random.seed(10)
doc = np.ones((800, 600), dtype=np.uint8) * 240   # white background
doc[50:70,   50:400] = 30    # heading line
doc[100:110, 50:350] = 40    # text line 1
doc[120:130, 50:380] = 40    # text line 2
doc[140:150, 50:300] = 40    # text line 3
doc[160:170, 50:360] = 40    # text line 4
noise = np.random.randint(0, 20, doc.shape, dtype=np.uint8)
doc   = cv2.add(doc, noise)

print(f"Document Shape : {doc.shape}")
print(f"Dtype          : {doc.dtype}")

# Step 1: Denoise
denoised = cv2.GaussianBlur(doc, (3, 3), 0)
print(f"After denoising - Mean pixel value : {denoised.mean():.2f}")

# Step 2: Simple Thresholding (binarize)
_, binary = cv2.threshold(denoised, 200, 255, cv2.THRESH_BINARY)
print(f"Binary image unique values : {np.unique(binary)}")

# Step 3: Adaptive Thresholding (handles uneven lighting better)
adaptive = cv2.adaptiveThreshold(
    denoised, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 11, 2
)
print(f"Adaptive threshold unique values : {np.unique(adaptive)}")

# Step 4: Morphological Opening (remove small noise)
kernel  = np.ones((2, 2), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
print(f"After morphological cleaning - non-zero pixels : {np.count_nonzero(cleaned)}")

# Step 5: Find contours (detect text regions)
contours, _ = cv2.findContours(
    cv2.bitwise_not(cleaned), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
print(f"Text regions detected : {len(contours)}")

print("\nDocument processing complete!")

# ─────────────────────────────────────────────────────────────
# SCENARIO 3: Smart Dashcam Lane Detection
# ─────────────────────────────────────────────────────────────
# A smart dashcam system needs to detect lane lines on the road
# to assist the driver or an autonomous vehicle.
# The pipeline:
#   1. Gaussian Blur to reduce road texture noise
#   2. Canny Edge Detection to find sharp boundaries
#   3. Region of Interest (ROI) masking — focus only on the road area
#   4. Hough Line Transform to detect straight lane lines
# This is the classic lane detection approach used in ADAS systems.
print("\n" + "=" * 60)
print("SCENARIO 3: Smart Dashcam Lane Detection")
print("Task: Canny Edge Detection + ROI Masking + Hough Line Transform")
print("=" * 60)

# Simulating a road image with two lane lines
np.random.seed(5)
road = np.zeros((480, 640), dtype=np.uint8)
road[200:, :] = 100   # road surface (gray)

# Left lane line (converging toward center)
for y in range(200, 480):
    x = int(200 - (y - 200) * 0.3)
    if 0 <= x < 640:
        road[y, max(0, x - 2):min(640, x + 2)] = 255

# Right lane line (converging toward center)
for y in range(200, 480):
    x = int(440 + (y - 200) * 0.3)
    if 0 <= x < 640:
        road[y, max(0, x - 2):min(640, x + 2)] = 255

noise = np.random.randint(0, 30, road.shape, dtype=np.uint8)
road  = cv2.add(road, noise)

print(f"Road image shape : {road.shape}")

# Step 1: Gaussian Blur
blurred = cv2.GaussianBlur(road, (5, 5), 0)

# Step 2: Canny Edge Detection
edges = cv2.Canny(blurred, 50, 150)
print(f"Edges detected : {np.count_nonzero(edges)} pixels")

# Step 3: Region of Interest (ROI) — focus on lower road area
roi_mask     = np.zeros_like(edges)
roi_vertices = np.array([[
    (0, 480), (0, 250), (320, 200), (640, 250), (640, 480)
]], dtype=np.int32)
cv2.fillPoly(roi_mask, roi_vertices, 255)
roi_edges = cv2.bitwise_and(edges, roi_mask)
print(f"Edges in ROI   : {np.count_nonzero(roi_edges)} pixels")

# Step 4: Hough Line Transform
lines = cv2.HoughLinesP(
    roi_edges,
    rho=1,
    theta=np.pi / 180,
    threshold=30,
    minLineLength=50,
    maxLineGap=20
)

if lines is not None:
    print(f"Lane lines detected : {len(lines)}")
    for i, line in enumerate(lines[:5]):
        x1, y1, x2, y2 = line[0]
        print(f"  Line {i + 1}: ({x1},{y1}) -> ({x2},{y2})")
else:
    print("No lane lines detected")

print("\nLane detection complete!")

print("\n" + "=" * 60)
print("DAY 12 - ALL SCENARIOS COMPLETE")
print("=" * 60)
