import cv2
import numpy as np

# Load the image
img = cv2.imread('parkingslots.jpg')
# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)

# Detect edges using Canny
edges = cv2.Canny(img_blur, 50, 150)

# Define parking spot coordinates (x1, y1, x2, y2)
parking_spots = [
    (31, 350, 127, 468),
    (245, 346, 369, 474),
    (470, 327, 599, 468),
    (703, 323, 816, 455),
    (920, 317, 1014, 463)
]

# Function to check if a parking spot is occupied
def check_occupancy(spot):
    x1, y1, x2, y2 = spot
    roi = edges[y1:y2, x1:x2]  # Region of Interest
    non_zero_count = cv2.countNonZero(roi)
    return non_zero_count > 500  # You can adjust the threshold

# Track available and occupied count
available_count = 0
occupied_count = 0

# Check and display each spot
for i, spot in enumerate(parking_spots):
    occupied = check_occupancy(spot)
    color = (0, 0, 255) if occupied else (0, 255, 0)  # Red or Green
    label = "Occupied" if occupied else "Empty"

    # Count the spots
    if occupied:
        occupied_count += 1
    else:
        available_count += 1

    # Draw rectangle and label
    cv2.rectangle(img, (spot[0], spot[1]), (spot[2], spot[3]), color, 2)
    text_pos = (spot[0], spot[1] - 10 if spot[1] - 10 > 10 else spot[1] + 20)
    cv2.putText(img, f"Spot {i+1}: {label}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# Print result to console
print(f"Total Parking Spots: {len(parking_spots)}")
print(f"Occupied Spots: {occupied_count}")
print(f"Available Spots: {available_count}")

# Show result
cv2.imshow('Parking Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()