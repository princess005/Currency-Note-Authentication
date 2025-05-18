import cv2
import numpy as np


# Load the note images
authentic_img = cv2.imread('authentic_note.jpg', 0)
test_img = cv2.imread('test_note.jpg', 0)

authentic_img=cv2.resize(authentic_img,(512,512))
test_img=cv2.resize(test_img,(512,512))

# ORB feature detector
orb = cv2.ORB_create()

# Detect and compute features
kp1, des1 = orb.detectAndCompute(authentic_img, None)
kp2, des2 = orb.detectAndCompute(test_img, None)

# Match features using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Count good matches
good_matches = [m for m in matches if m.distance < 50]
print("Number of good matches:", len(good_matches))

# Threshold to decide authenticity
if len(good_matches) > 15:
    print("Result: GENUINE")
else:
    print("Result: COUNTERFEIT")

# Display matches
result = cv2.drawMatches(authentic_img, kp1, test_img, kp2, matches[:20], None, flags=2)
cv2.imshow('Feature Matches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()