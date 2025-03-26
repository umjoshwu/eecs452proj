import cv2
import numpy as np

def detect_ir_camera_spots(frame1, frame2, 
                            diff_threshold=25, 
                            white_threshold=200, 
                            min_blob_area=10, 
                            max_blob_area=500):
    """
    Detect differences between two frames, focusing on potential IR camera spots.
    
    Parameters:
    - frame1: First input frame
    - frame2: Second input frame
    - diff_threshold: Threshold for pixel intensity difference
    - white_threshold: Threshold for white intensity
    - min_blob_area: Minimum area of detected blobs
    - max_blob_area: Maximum area of detected blobs
    
    Returns:
    - Difference image with detected blobs highlighted
    - List of detected blob information
    """
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference between frames
    frame_diff = cv2.absdiff(gray1, gray2)
    
    # Apply threshold to difference image
    _, diff_thresh = cv2.threshold(frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
    
    # Find contours in the difference image
    contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the first frame to draw results
    result_frame = frame1.copy()
    
    # List to store detected blob information
    detected_blobs = []
    
    # Process each contour
    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)
        
        # Filter contours by area
        if min_blob_area < area < max_blob_area:
            # Get moments to calculate center
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Get bounding circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            
            # Check if the blob is predominantly white in the original image
            mask = np.zeros(gray1.shape, np.uint8)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            mean_intensity = cv2.mean(gray2, mask=mask)[0]
            
            # If the blob is bright in the second image
            if mean_intensity > white_threshold:
                blob_info = {
                    'center': (cx, cy),
                    'area': area,
                    'radius': radius,
                    'circularity': circularity,
                    'mean_intensity': mean_intensity
                }
                detected_blobs.append(blob_info)
                
                # Draw the blob on the result frame
                cv2.drawContours(result_frame, [contour], 0, (0, 255, 0), 2)
                cv2.circle(result_frame, (cx, cy), 3, (0, 0, 255), -1)
    
    return result_frame, detected_blobs

def main():
    # Load two frames for comparison
    frame1 = cv2.imread('pics4.jpg')  # Frame without IR light
    frame2 = cv2.imread('pics1.jpg')     # Frame with IR light
    
    if frame1 is None or frame2 is None:
        print("Error: Could not read images")
        return
    
    # Ensure frames are the same size
    print(frame1.shape, frame2.shape)

    if frame1.shape != frame2.shape:
        print("Error: Frames must be the same size")
        return
    
    try:
        # Detect potential IR camera spots
        result_frame, detected_blobs = detect_ir_camera_spots(
            frame1, frame2, 
            diff_threshold=25,
            white_threshold=200,
            min_blob_area=10,
            max_blob_area=500
        )
        
        # Print detected blob information
        print(f"Number of potential IR camera spots detected: {len(detected_blobs)}")
        for i, blob in enumerate(detected_blobs, 1):
            print(f"\nBlob {i}:")
            print(f"  Center: {blob['center']}")
            print(f"  Area: {blob['area']:.2f}")
            print(f"  Radius: {blob['radius']:.2f}")
            print(f"  Circularity: {blob['circularity']:.2f}")
            print(f"  Mean Intensity: {blob['mean_intensity']:.2f}")
        
        # Display the result
        cv2.imshow('Potential IR Camera Spots', result_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
