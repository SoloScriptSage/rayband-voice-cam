"""
Standalone test for hand detection.
Run this to test hand detection independently.
"""

import cv2
import numpy as np

def detect_hands(frame):
    """Simple hand detection test."""
    # Convert to HSV and YCrCb
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    
    # Skin color ranges
    lower_hsv = np.array([0, 20, 70], dtype=np.uint8)
    upper_hsv = np.array([20, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    
    # Combine masks
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    
    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area
    hand_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            hand_contours.append(contour)
    
    return hand_contours, mask

def main():
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    print("Camera opened successfully!")
    print("Show your hand to the camera")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read frame")
            break
        
        # Detect hands
        hand_contours, mask = detect_hands(frame)
        
        # Draw results
        if hand_contours:
            print(f"Found {len(hand_contours)} hands")
            cv2.drawContours(frame, hand_contours, -1, (0, 255, 0), 3)
            
            for contour in hand_contours:
                # Draw convex hull
                hull = cv2.convexHull(contour)
                cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)
                
                # Find center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)
                    cv2.putText(frame, "HAND", (cx-30, cy-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show info
        cv2.putText(frame, f"Hands detected: {len(hand_contours)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frames
        cv2.imshow("Hand Detection Test", frame)
        cv2.imshow("Mask (Skin Detection)", mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test completed!")

if __name__ == "__main__":
    main()