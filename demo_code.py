import cv2
import numpy as np

# Glare detection parameters
GLARE_THRESHOLD = 200    # Brightness threshold (0-255)
MIN_AREA = 500           # Minimum glare region area (pixels)

class Config:
    # Basic glare detection
    GLARE_THRESHOLD = 200
    MIN_GLARE_AREA = 500
    
    # Lens detection
    CIRCLE_DP = 2          # Hough circle detection sensitivity (smaller = more sensitive)
    MIN_RADIUS = 10        # Minimum circle radius
    MAX_RADIUS = 100       # Maximum circle radius
    EDGE_THRESHOLD = 30    # Edge detection threshold
    CIRCLE_SCORE = 0.7     # Shape matching threshold (0-1)

def detect_glare(frame, threshold=GLARE_THRESHOLD, min_area=MIN_AREA):
    """
    Core algorithm for glare detection
    :param frame: Input image frame (BGR format)
    :param threshold: Brightness threshold
    :param min_area: Minimum glare region area
    :return: (has_glare, processed_frame)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    
    # Perform thresholding
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    has_glare = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            has_glare = True
            # Draw bounding box around glare region
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    return has_glare, frame

def detect_lens(frame):
    """
    Detect lens circular reflection
    Returns: (lens_detected, processed_frame)
    """
    # Preprocessing: detect within glare regions
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (15,15), 0)
    
    # Hough Circle detection
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=Config.CIRCLE_DP,
        minDist=50,
        param1=Config.EDGE_THRESHOLD,
        param2=30,
        minRadius=Config.MIN_RADIUS,
        maxRadius=Config.MAX_RADIUS
    )
    
    has_lens = False
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            # Verify circular features
            x, y, r = circle
            roi = frame[y-r:y+r, x-r:x+r]
            
            # Feature verification (color uniformity + edge strength)
            if verify_lens_features(roi, r):
                cv2.circle(frame, (x,y), r, (0,255,0), 3)
                cv2.putText(frame, 'LENS', (x-r, y-r-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                has_lens = True
    
    return has_lens, frame

def verify_lens_features(roi, radius):
    """
    Verify lens characteristics:
    1. Color uniformity (difference between center and edge)
    2. Circular edge strength
    """
    if roi.size == 0:
        return False
    
    # Create circular mask
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (radius,radius), radius, 255, -1)
    
    # Calculate center-edge difference
    mean_center = cv2.mean(roi, mask=cv2.erode(mask, None, iterations=2))[0]
    mean_edge = cv2.mean(roi, mask=mask - cv2.erode(mask, None, iterations=2))[0]
    
    # Color difference threshold (typical lens reflection has brighter center)
    if (mean_center - mean_edge) < 50:
        return False
    
    # Edge strength verification
    edges = cv2.Canny(roi, 100, 200)
    edge_score = cv2.mean(edges, mask=mask)[0]
    
    return edge_score > Config.CIRCLE_SCORE * 255

def main():
    cap = cv2.VideoCapture(1)
    
    # Create control panel
    cv2.namedWindow('Control Panel')
    cv2.createTrackbar('Threshold', 'Control Panel', Config.GLARE_THRESHOLD, 255, lambda x: None)
    cv2.createTrackbar('Min Radius', 'Control Panel', Config.MIN_RADIUS, 100, lambda x: None)
    cv2.createTrackbar('Max Radius', 'Control Panel', Config.MAX_RADIUS, 200, lambda x: None)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update parameters
        Config.GLARE_THRESHOLD = cv2.getTrackbarPos('Threshold', 'Control Panel')
        Config.MIN_RADIUS = cv2.getTrackbarPos('Min Radius', 'Control Panel')
        Config.MAX_RADIUS = cv2.getTrackbarPos('Max Radius', 'Control Panel')
        
        # Detection processing
        _, glare_frame = detect_glare(frame)
        has_lens, result_frame = detect_lens(glare_frame)
        
        # Display status
        status = f"Lens: {'Found' if has_lens else 'Searching'}"
        cv2.putText(result_frame, status, (10,60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        
        cv2.imshow('Lens Detection', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()