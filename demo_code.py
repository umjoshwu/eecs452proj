import cv2
import numpy as np
import math

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

def detect_lens_flare(frame):
    """
    Detect lens flare artifacts 
    Returns: (has_flare, processed_frame)
    """
    class Config:
        LENS_FLARE_BRIGHTNESS = 200
        GAMMA_VALUE = 20

    # State persistence initialization
    if not hasattr(detect_lens_flare, "last_flare"):
        detect_lens_flare.last_flare = False
        detect_lens_flare.has_new_ellipse = False  # New state flag

    # Detection logic
    current_flare = False
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    _, l, _ = cv2.split(hls)
    
    if np.mean(l) > Config.LENS_FLARE_BRIGHTNESS:
        cv2.putText(frame, "High Brightness", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        current_flare = True
    else:
        # Ellipse detection logic
        blur = cv2.medianBlur(frame, 3)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
        
        gamma = Config.GAMMA_VALUE
        lookup_table = np.array([np.clip(pow(i/255.0, gamma)*255.0, 0, 255) 
                               for i in range(256)], dtype=np.uint8)
        gamma_gray = cv2.LUT(gray, lookup_table)
        
        _, thresh = cv2.threshold(gamma_gray, 0, 255, 
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        edges = cv2.Canny(thresh, 600, 100, 3)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if len(cnt) > 150:
                area = cv2.contourArea(cnt)
                if area < 100:
                    continue
                
                try:
                    ell = cv2.fitEllipse(cnt)
                except:
                    continue
                
                (x, y), (ma, MA), angle = ell
                if ma == 0 or MA == 0:
                    continue
                
                ellipse_area = math.pi * (ma/2) * (MA/2)
                if ellipse_area == 0:
                    continue
                
                area_ratio = area / ellipse_area
                axis_ratio = max(ma, MA) / min(ma, MA)
                
                if area_ratio > 0.2 and axis_ratio < 2:
                    cv2.ellipse(frame, ell, (0, 255, 255), 2)
                    current_flare = True
                    detect_lens_flare.has_new_ellipse = True  # Mark new detection

    # State persistence logic
    if detect_lens_flare.has_new_ellipse:
        detect_lens_flare.last_flare = current_flare
        detect_lens_flare.has_new_ellipse = False
    elif current_flare:
        detect_lens_flare.last_flare = True
        detect_lens_flare.has_new_ellipse = True

    return detect_lens_flare.last_flare, frame
    
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
        has_glare, glare_frame = detect_glare(frame)
        has_lens, lens_frame = detect_lens(glare_frame)
        has_flare, result_frame = detect_lens_flare(lens_frame)
        
        # Display status
        cv2.putText(result_frame, f"Glare: {'Yes' if has_glare else 'No'}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_frame, f"Lens: {'Found' if has_lens else 'Searching'}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(result_frame, f"Flare: {'Detected' if has_flare else 'Clear'}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Lens Detection', result_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
