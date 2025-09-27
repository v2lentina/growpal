#!/usr/bin/env python3

import cv2
import time
from ultralytics import YOLO
import numpy as np
from collections import Counter
import os

def detect_purple_flowers(image):
    """
    Detects purple/violet flowers that might be misclassified as weeds
    Returns: (is_purple_flower, purple_ratio)
    """
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Purple/violet color ranges in HSV
    lower_purple1 = np.array([115, 50, 50])   
    upper_purple1 = np.array([135, 255, 255]) 
    lower_purple2 = np.array([140, 50, 50])   
    upper_purple2 = np.array([160, 255, 255]) 
    
    # Create masks for purple colors
    mask1 = cv2.inRange(hsv, lower_purple1, upper_purple1)
    mask2 = cv2.inRange(hsv, lower_purple2, upper_purple2)
    purple_mask = cv2.bitwise_or(mask1, mask2)
    
    # Count purple pixels
    purple_pixels = cv2.countNonZero(purple_mask)
    total_pixels = image.shape[0] * image.shape[1]
    purple_ratio = purple_pixels / total_pixels
    
    # If more than 3% of image is purple, it's likely a flower
    return purple_ratio > 0.03, purple_ratio

def analyze_detection_context(image, bbox):
    """
    Analyzes the context around a detection to determine if it's likely a weed or flower
    bbox format: [x, y, width, height]
    """
    x, y, w, h = bbox
    
    # Extract the region of interest (with some padding)
    pad = 20
    roi_x1 = max(0, x - pad)
    roi_y1 = max(0, y - pad)
    roi_x2 = min(image.shape[1], x + w + pad)
    roi_y2 = min(image.shape[0], y + h + pad)
    
    roi = image[roi_y1:roi_y2, roi_x1:roi_x2]
    
    if roi.size == 0:
        return False, 0.0
    
    # Check for purple colors in the detection area
    is_purple, purple_ratio = detect_purple_flowers(roi)
    
    # Calculate size - flowers tend to be larger than typical weeds
    detection_area = w * h
    relative_size = detection_area / (image.shape[0] * image.shape[1])
    
    return is_purple, purple_ratio, relative_size

class WeedDetector:
    def __init__(self, model_path="best.pt", confidence_threshold=0.35):
   
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.camera = None
        
        self.class_names = {0: 'weed', 1: 'crop'} 
        
        self.verification_attempts = 3  
        self.verification_delay = 0.5  
        self.min_confidence_for_action = 0.7 
        
        print(f"WeedDetector initialised")
        
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
                
            self.model = YOLO(self.model_path)
            print(f"Model successfully loaded: {self.model_path}")
            return True
        except Exception as e:
            print(f"Error while loading model: {e}")
            return False
    
    def init_camera(self, camera_index=0):
        try:
            self.camera = cv2.VideoCapture(camera_index)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            #Test
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Can't capture image from camera")
                
            print(f"Camera initialized (Index: {camera_index})")
            return True
        except Exception as e:
            print(f"Error while initializing camera: {e}")
            return False
    
    def take_photo(self):
        if self.camera is None:
            print("Camera not initialized!")
            return None
            
        ret, frame = self.camera.read()
        if ret:
            return frame
        else:
            print("Error capturing image from camera")
            return None
    
    def detect_objects_in_image(self, image):
        """
        Detects objects in an image with smart filtering to reduce false positives
        """
        if self.model is None:
            print("❌ Model not loaded!")
            return []
            
        results = self.model(image, conf=self.confidence_threshold)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Class and Confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = self.class_names.get(class_id, f"unknown_{class_id}")
                    
                    # Bounding Box Coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = [int(x1), int(y1), int(x2-x1), int(y2-y1)]  # [x, y, width, height]
                    
                    # 🔍 SMART FILTERING - Check if this is likely a flower misclassified as weed
                    if class_name == 'weed':
                        is_purple, purple_ratio, relative_size = analyze_detection_context(image, bbox)
                        
                        # Filter purple flowers (keep this strict)
                        if is_purple and purple_ratio > 0.02:
                            print(f"   🌸 Filtering out purple flower (purple ratio: {purple_ratio:.3f})")
                            continue  # Skip this detection
                        
                        # More lenient size filtering - only filter VERY large objects
                        if relative_size > 0.15:  # More than 15% of image (was 8%)
                            print(f"   🌺 Filtering out very large object (size ratio: {relative_size:.3f})")
                            continue  # Skip this detection
                        
                        # Filter high-confidence flower-like detections (more aggressive for high confidence)
                        if confidence > 0.5 and (purple_ratio > 0.005 or relative_size > 0.06):
                            print(f"   🌻 Filtering out high-confidence flower-like detection (conf: {confidence:.1%})")
                            continue
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': bbox,
                        'center': [int((x1+x2)/2), int((y1+y2)/2)]  # Center point
                    })
        
        return detections
    
    def verify_weed_detection(self):
        """
        Multi-step verification: Takes multiple images and checks for consistent weed detection
        
        Returns:
            dict: {'weed_confirmed': bool, 'position': [x,y], 'confidence': float}
        """
        print(f"Starting weed verification ({self.verification_attempts} attempts)...")
        
        weed_detections = []
        all_attempts = []
        
        for attempt in range(self.verification_attempts):
            print(f"   Attempt {attempt + 1}/{self.verification_attempts}")
            
            # Take photo
            image = self.take_photo()
            if image is None:
                continue
                
            # Detect objects
            detections = self.detect_objects_in_image(image)
            all_attempts.append(detections)

            # Only collect weed detections
            weeds_in_this_image = [d for d in detections if d['class'] == 'weed']
            weed_detections.extend(weeds_in_this_image)

            print(f"      → {len(weeds_in_this_image)} weeds found, {len([d for d in detections if d['class'] == 'crop'])} crops")

            time.sleep(self.verification_delay)

        # Evaluation
        result = self._evaluate_detections(weed_detections, all_attempts)
        return result
    
    def _evaluate_detections(self, weed_detections, all_attempts):
        """Evaluates the collected detections"""
        total_attempts = len(all_attempts)
        
        if len(weed_detections) == 0:
            print("No weeds detected in any attempt")
            return {
                'weed_confirmed': False,
                'position': None,
                'confidence': 0.0,
                'details': f"0/{total_attempts} attempts showed weeds"
            }

        # At least 2 out of 3 attempts should detect weeds
        attempts_with_weed = sum(1 for attempt in all_attempts if any(d['class'] == 'weed' for d in attempt))
        confidence_threshold_for_action = 0.6  # 60% of attempts
        
        if attempts_with_weed / total_attempts >= confidence_threshold_for_action:
            # Average position and confidence calculation
            avg_x = sum(w['center'][0] for w in weed_detections) / len(weed_detections)
            avg_y = sum(w['center'][1] for w in weed_detections) / len(weed_detections)
            avg_confidence = sum(w['confidence'] for w in weed_detections) / len(weed_detections)

            print(f"WEED CONFIRMED! Position: ({avg_x:.0f}, {avg_y:.0f})")
            return {
                'weed_confirmed': True,
                'position': [int(avg_x), int(avg_y)],
                'confidence': avg_confidence,
                'details': f"{attempts_with_weed}/{total_attempts} attempts showed weeds"
            }
        else:
            print(f"WEED UNCERTAIN - only detected in {attempts_with_weed}/{total_attempts} attempts")
            return {
                'weed_confirmed': False,
                'position': None,
                'confidence': 0.0,
                'details': f"Only {attempts_with_weed}/{total_attempts} attempts showed weeds"
            }
    
    def scan_for_weeds(self):
        """
        Main function for PiCrawler: Scans for weeds and provides action recommendation
        
        Returns:
            dict: {'action': 'pull_weed'/'continue', 'target_position': [x,y], 'confidence': float}
        """
        print("Starting weed scan...")
        
        if self.model is None:
            print("Model not loaded! Call load_model() first")
            return {'action': 'error', 'message': 'Model not loaded'}
            
        if self.camera is None:
            print("Camera not initialized! Call init_camera() first")
            return {'action': 'error', 'message': 'Camera not initialized'}

        # Multiple verification
        result = self.verify_weed_detection()
        
        if result['weed_confirmed']:
            return {
                'action': 'pull_weed',
                'target_position': result['position'],
                'confidence': result['confidence'],
                'details': result['details']
            }
        else:
            return {
                'action': 'continue',
                'target_position': None,
                'confidence': result['confidence'],
                'details': result['details']
            }
    
    def cleanup(self):
        """Cleans up resources"""
        if self.camera is not None:
            self.camera.release()
            print("Camera released")

# Test functions (without real camera)
def test_with_image_file(image_path):
    """Tests the system with an image file"""
    detector = WeedDetector()
    
    if not detector.load_model():
        return

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    print(f"Analyzing image: {image_path}")
    detections = detector.detect_objects_in_image(image)

    print(f"Detections:")
    for det in detections:
        print(f"   - {det['class']}: {det['confidence']:.2f} at position {det['center']}")

# Main program
if __name__ == "__main__":
    print("🌱 PiCrawler Weed Detector started")

    detector = WeedDetector()

    # Load model
    if not detector.load_model():
        print("Program terminated - Model could not be loaded")
        exit(1)

    # Initialize camera
    if not detector.init_camera():
        print("Program terminated - Camera could not be initialized")
        exit(1)
    
    try:
        # Main loop (can be called later by PiCrawler)
        print("\nReady for weed detection! Press Ctrl+C to exit")

        while True:
            input("Press Enter for weed scan...")
            
            result = detector.scan_for_weeds()
            print(f"\n📋 Result: {result}")

            if result['action'] == 'pull_weed':
                print(f"ACTION: Robot should move to position {result['target_position']} and remove weed!")
            elif result['action'] == 'continue':
                print("All clean - Robot can continue moving")

            print("-" * 50)
            
    except KeyboardInterrupt:
        print("\nProgram terminated")
    finally:
        detector.cleanup()
