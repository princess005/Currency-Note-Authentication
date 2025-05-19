import cv2
import numpy as np
import math

def preprocess_frame(frame):
    frame = cv2.resize(frame, (400, 400))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return frame, thresh

def find_largest_contour(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    return max_contour

def count_defects(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 3:
        return 0

    defects = cv2.convexityDefects(contour, hull)
    if defects is None:
        return 0

    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = np.linalg.norm(np.array(end) - np.array(start))
        b = np.linalg.norm(np.array(far) - np.array(start))
        c = np.linalg.norm(np.array(end) - np.array(far))

        angle = np.arccos((b**2 + c**2 - a**2)/(2*b*c)) * 57
        if angle <= 90:
            count += 1
            cv2.circle(drawing, far, 5, [0, 0, 255], -1)
    return count

def classify_gesture(defect_count):
    if defect_count == 0:
        return "Fist ✊"
    elif defect_count == 1:
        return "One Finger ☝️"
    elif defect_count == 2:
        return "Peace ✌️"
    elif defect_count == 3:
        return "Call Me 🤙"
    elif defect_count == 4:
        return "Thumbs Up 👍"
    elif defect_count >= 5:
        return "Open Hand 🖐️"
    else:
        return "Unknown Gesture"

def run_webcam():
    cap = cv2.VideoCapture(0)
    capture_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame, thresh = preprocess_frame(frame)
        contour = find_largest_contour(thresh)
        drawing = np.zeros_like(frame)

        if contour is not None and cv2.contourArea(contour) > 1000:
            cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 2)
            defects = count_defects(contour, drawing)
            gesture = classify_gesture(defects)
            cv2.putText(drawing, f'Gesture: {gesture}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            gesture = "No hand"

        # Show windows
        cv2.imshow("Threshold", thresh)
        cv2.imshow("Gesture", drawing)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save gesture image
            filename = f"gesture_capture_{capture_count}.png"
            cv2.imwrite(filename, drawing)
            print(f"[INFO] Gesture captured and saved as {filename}")
            capture_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Auto-run without needing if __name__ block (optional)
run_webcam()
