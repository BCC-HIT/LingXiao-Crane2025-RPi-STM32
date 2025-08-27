#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO Recognition Module - Intelligent Delivery Robot Vision Recognition System
New Features:
1. Configurable inference resolution (separate settings for shelf and area)
2. Simplified position mapping algorithm (based on absolute position)
3. More flexible configuration options
4. Support for both PyTorch (.pt) and ONNX (.onnx) model formats
"""

from ultralytics import YOLO
import torch
import cv2
import numpy as np
import json
import os
import time
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter, defaultdict
from datetime import datetime


class YOLOModule:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize YOLO recognition module

        Args:
            config_path: Configuration file path
        """
        self.config = self.load_config(config_path)
        self.shelf_model = None
        self.area_model = None
        self.device = None

        # Inference resolution settings
        self.shelf_inference_size = self.config["yolo"].get("shelf_inference_size", 640)
        self.area_inference_size = self.config["yolo"].get("area_inference_size", 640)
        self.position_mapping_method = "absolute"

        # Initialize models
        self.load_models()

    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        default_config = {
            "yolo": {
                "yolo_path": "~/yolov5",
                "shelf_model_path": "~/models/front02_yolov5s/weights/best_linux.pt",
                "area_model_path": "~/models/yolov5s_backdatasets_exp1/weights/best_linux.pt",
                "shelf_confidence_threshold": 0.85,
                "area_confidence_threshold": 0.6,
                "nms_threshold": 0.4,
                "consensus_method": "voting",
                "min_consensus_count": 3,
                "device": "cpu",
                "shelf_inference_size": 640,
                "area_inference_size": 640,

                "_comment_position_mapping": "Position mapping parameter settings - based on absolute position judgment",
                "shelf_position_mapping": {
                    "y_upper_lower_split": 0.55,
                    "x_left_split": 0.33,
                    "x_right_split": 0.67
                },
                "area_position_mapping": {
                    "x_splits": [0.167, 0.333, 0.5, 0.667, 0.833]
                }
            }
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Merge default configuration
                    if 'yolo' not in config:
                        config['yolo'] = default_config['yolo']
                    else:
                        for key, value in default_config['yolo'].items():
                            if key not in config['yolo']:
                                config['yolo'][key] = value
                    return config
            else:
                # Create default configuration file
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4, ensure_ascii=False)
                print(f"Default configuration file created: {config_path}")
                return default_config
        except Exception as e:
            print(f"Failed to load configuration file, using default settings: {e}")
            return default_config

    def load_models(self):
        """Load YOLO models (supports both PyTorch .pt and ONNX .onnx formats)"""
        try:
            yolo_config = self.config["yolo"]
            # Get device information directly from configuration
            self.device = yolo_config.get("device", "cpu")

            # Load shelf model
            shelf_model_path = os.path.expanduser(yolo_config["shelf_model_path"])
            if os.path.exists(shelf_model_path):
                print(f"Loading shelf model: {shelf_model_path}")
                
                # Check model format and load accordingly
                if shelf_model_path.lower().endswith('.onnx'):
                    # For ONNX models, specify task explicitly
                    self.shelf_model = YOLO(shelf_model_path, task='detect')
                    print(f"Loaded ONNX shelf model on device: {self.device}")
                else:
                    # For PyTorch models (.pt)
                    self.shelf_model = YOLO(shelf_model_path)
                    self.shelf_model.to(self.device)
                    print(f"Loaded PyTorch shelf model on device: {self.device}")

                # Configure model parameters
                self.shelf_model.conf = yolo_config.get("shelf_confidence_threshold", 0.85)
                self.shelf_model.iou = yolo_config["nms_threshold"]
                print(f"Shelf model loaded successfully")
            else:
                print(f"ERROR: Shelf model file not found: {shelf_model_path}")
                self.shelf_model = None

            # Load area model
            area_model_path = os.path.expanduser(yolo_config["area_model_path"])
            if os.path.exists(area_model_path):
                print(f"Loading area model: {area_model_path}")
                
                # Check model format and load accordingly
                if area_model_path.lower().endswith('.onnx'):
                    # For ONNX models, specify task explicitly
                    self.area_model = YOLO(area_model_path, task='detect')
                    print(f"Loaded ONNX area model on device: {self.device}")
                else:
                    # For PyTorch models (.pt)
                    self.area_model = YOLO(area_model_path)
                    self.area_model.to(self.device)
                    print(f"Loaded PyTorch area model on device: {self.device}")

                # Configure model parameters
                self.area_model.conf = yolo_config.get("area_confidence_threshold", 0.6)
                self.area_model.iou = yolo_config["nms_threshold"]
                print(f"Area model loaded successfully")
            else:
                print(f"ERROR: Area model file not found: {area_model_path}")
                self.area_model = None

        except Exception as e:
            print(f"ERROR: Failed to load YOLO models: {e}")
            print(f"Troubleshooting tips:")
            print(f"  1. For ONNX models: Ensure ultralytics supports ONNX inference")
            print(f"  2. For PyTorch models: Ensure model file is valid .pt format")
            print(f"  3. Check device compatibility (CPU/GPU)")
            print(f"  4. Verify model file permissions and accessibility")
            self.shelf_model = None
            self.area_model = None

    def detect_single_image(self, image: np.ndarray, model: Any, model_type: str, inference_size: int) -> List[Dict]:
        """
        Single image detection (supports both PyTorch and ONNX models)
        """
        if model is None:
            return []

        try:
            # YOLO inference - for both PyTorch and ONNX models
            # ONNX models will automatically use the appropriate backend
            results = model.predict(
                source=image, 
                imgsz=inference_size, 
                conf=model.conf, 
                iou=model.iou,
                device=self.device, 
                verbose=False
            )

            # results is a list, for single image prediction, we only take the first element
            result = results[0]

            detections = []
            # Iterate through detection boxes
            for box in result.boxes:
                # Get coordinates, confidence and class ID
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                class_name = result.names[cls]  # Get class name from results
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Build dictionary completely compatible with old code to ensure subsequent functions need no changes
                detection_info = {
                    'class': class_name,
                    'confidence': conf,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'center': [float(center_x), float(center_y)],
                    'model_type': model_type,
                    'inference_size': inference_size
                }
                detections.append(detection_info)

            return detections

        except Exception as e:
            print(f"ERROR: {model_type} model inference failed: {e}")
            print(f"Model type: {'ONNX' if str(model.model).endswith('.onnx') else 'PyTorch'}")
            return []

    def get_model_info(self, model, model_path: str) -> str:
        """Get model format information"""
        if model is None:
            return "Not loaded"
        
        if model_path.lower().endswith('.onnx'):
            return f"ONNX model"
        elif model_path.lower().endswith('.pt'):
            return f"PyTorch model"
        else:
            return f"Unknown format"

    def map_shelf_position(self, detection: Dict, image_shape: Tuple[int, int],
                           all_detections: List[Dict] = None) -> int:
        """
        Shelf position mapping (based on absolute position)

        Args:
            detection: Current detection result
            image_shape: Image dimensions (height, width)
            all_detections: Unused, kept for interface compatibility

        Returns:
            Shelf position (1-6), 0 indicates mapping failure
        """
        try:
            center_x, center_y = detection['center']
            img_height, img_width = image_shape

            # Normalized coordinates
            norm_x = center_x / img_width
            norm_y = center_y / img_height

            # Get configured threshold parameters
            shelf_config = self.config["yolo"].get("shelf_position_mapping", {})

            # Y coordinate threshold: divide upper and lower layers
            y_split = shelf_config.get("y_upper_lower_split", 0.55)

            # X coordinate thresholds: divide left, center, right
            x_left_split = shelf_config.get("x_left_split", 0.33)
            x_right_split = shelf_config.get("x_right_split", 0.67)

            # Position mapping based on image analysis
            if norm_y < y_split:  # Upper layer
                if norm_x < x_left_split:
                    return 1  # Upper left is position 1
                elif norm_x < x_right_split:
                    return 2  # Upper center is position 2
                else:
                    return 3  # Upper right is position 3
            else:  # Lower layer
                if norm_x < x_left_split:
                    return 4  # Lower left is position 4
                elif norm_x < x_right_split:
                    return 5  # Lower center is position 5
                else:
                    return 6  # Lower right is position 6

        except Exception as e:
            print(f"Shelf position mapping failed: {e}")
            return 0

    def map_area_position(self, detection: Dict, image_shape: Tuple[int, int],
                          all_detections: List[Dict] = None) -> str:
        """
        Area position mapping (based on absolute position)

        Args:
            detection: Current detection result
            image_shape: Image dimensions (height, width)
            all_detections: Unused, kept for interface compatibility

        Returns:
            Area position (a-f), empty string indicates mapping failure
        """
        try:
            center_x, center_y = detection['center']
            img_height, img_width = image_shape

            # Normalized X coordinate
            norm_x = center_x / img_width

            # Get configured threshold parameters
            area_config = self.config["yolo"].get("area_position_mapping", {})

            # Area position dividing points (5 dividing points, 6 areas)
            # Default equal division, but can be adjusted according to actual situation
            splits = area_config.get("x_splits", [0.2, 0.35, 0.5, 0.65, 0.8])

            # Determine position based on X coordinate
            position_map = ['a', 'b', 'c', 'd', 'e', 'f']

            for i, split in enumerate(splits):
                if norm_x < split:
                    return position_map[i]

            # If no match, return the last position
            return position_map[-1]

        except Exception as e:
            print(f"Area position mapping failed: {e}")
            return ''

    def detect_shelf_numbers(self, images: List[np.ndarray]) -> Dict:
        """
        Recognize shelf numbers (using configurable inference resolution)
        """
        if self.shelf_model is None:
            return {'success': False, 'error': 'Shelf model not loaded'}

        try:
            all_detections = []

            print(f"Recognizing {len(images)} shelf images...")
            print(f"Using inference resolution: {self.shelf_inference_size}")

            # Detect each image
            for i, image in enumerate(images):
                detections = self.detect_single_image(image, self.shelf_model, "shelf", self.shelf_inference_size)
                print(f"  Image {i + 1}: Detected {len(detections)} objects")

                # Add position information for each detection result
                for detection in detections:
                    position = self.map_shelf_position(detection, image.shape[:2], detections)
                    if position > 0:  # Valid position
                        detection['position'] = position
                        detection['number'] = int(detection['class'])  # Class is the number
                        detection['image_index'] = i
                        detection['image_height'] = image.shape[0]
                        detection['image_width'] = image.shape[1]
                        all_detections.append(detection)

                        # Display coordinate information for debugging
                        norm_x = detection['center'][0] / image.shape[1]
                        norm_y = detection['center'][1] / image.shape[0]
                        print(
                            f"      Image {i + 1}: Position {position} -> Number {detection['number']} (Confidence: {detection['confidence']:.3f}) Coords: ({norm_x:.3f}, {norm_y:.3f}) @{self.shelf_inference_size}")

            # Get consensus results
            consensus_result = self.get_consensus_result(all_detections, "shelf")

            return {
                'success': True,
                'detections': all_detections,
                'consensus': consensus_result,
                'inference_size': self.shelf_inference_size,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }

        except Exception as e:
            print(f"Shelf recognition failed: {e}")
            return {'success': False, 'error': str(e)}

    def detect_area_numbers(self, images: List[np.ndarray]) -> Dict:
        """
        Recognize area numbers (using configurable inference resolution)
        """
        if self.area_model is None:
            return {'success': False, 'error': 'Area model not loaded'}

        try:
            all_detections = []

            print(f"Recognizing {len(images)} area images...")
            print(f"Using inference resolution: {self.area_inference_size}")

            # Detect each image
            for i, image in enumerate(images):
                detections = self.detect_single_image(image, self.area_model, "area", self.area_inference_size)
                print(f"  Image {i + 1}: Detected {len(detections)} objects")

                # Add position information for each detection result
                for detection in detections:
                    position = self.map_area_position(detection, image.shape[:2], detections)
                    if position:  # Valid position
                        detection['position'] = position

                        # Parse number
                        class_name = detection['class']
                        if class_name == 'O':
                            detection['number'] = 0  # Empty position
                        else:
                            # Extract number from '1#', '2#' etc. format
                            detection['number'] = int(class_name.replace('#', ''))

                        detection['image_index'] = i
                        detection['image_height'] = image.shape[0]
                        detection['image_width'] = image.shape[1]
                        all_detections.append(detection)

                        # Display coordinate information for debugging
                        norm_x = detection['center'][0] / image.shape[1]
                        norm_y = detection['center'][1] / image.shape[0]
                        number_str = "Empty" if detection['number'] == 0 else f"Number {detection['number']}"
                        print(
                            f"      Image {i + 1}: Position {position} -> {number_str} (Confidence: {detection['confidence']:.3f}) Coords: ({norm_x:.3f}, {norm_y:.3f}) @{self.area_inference_size}")

            # Get consensus results
            consensus_result = self.get_consensus_result(all_detections, "area")

            return {
                'success': True,
                'detections': all_detections,
                'consensus': consensus_result,
                'inference_size': self.area_inference_size,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }

        except Exception as e:
            print(f"Area recognition failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_consensus_result(self, detections: List[Dict], detection_type: str) -> Dict:
        """Get consensus results from multiple detections"""
        config = self.config["yolo"]
        method = config["consensus_method"]
        min_count = config["min_consensus_count"]

        # Group by position
        position_groups = defaultdict(list)
        for detection in detections:
            position = detection['position']
            position_groups[position].append(detection)

        consensus_results = {}

        for position, group_detections in position_groups.items():
            if len(group_detections) < min_count:
                print(f"  Position {position}: Insufficient detection count ({len(group_detections)} < {min_count})")
                continue

            if method == "voting":
                # Voting method: select the number with the most occurrences
                number_votes = Counter(d['number'] for d in group_detections)
                most_common = number_votes.most_common(1)
                if most_common:
                    number, count = most_common[0]
                    if count >= min_count:
                        # Calculate average confidence for this number
                        same_number_detections = [d for d in group_detections if d['number'] == number]
                        avg_confidence = np.mean([d['confidence'] for d in same_number_detections])

                        consensus_results[position] = {
                            'number': number,
                            'confidence': float(avg_confidence),
                            'vote_count': count,
                            'total_detections': len(group_detections)
                        }

            elif method == "confidence":
                # Confidence method: select the detection result with highest confidence
                best_detection = max(group_detections, key=lambda d: d['confidence'])
                consensus_results[position] = {
                    'number': best_detection['number'],
                    'confidence': best_detection['confidence'],
                    'vote_count': 1,
                    'total_detections': len(group_detections)
                }

        return consensus_results

    def format_result_string(self, shelf_results: Dict, area_results: Dict) -> str:
        """Format recognition results as string"""
        result_parts = []

        # Process shelf results (positions 1-6)
        shelf_consensus = {}
        if shelf_results.get('success') and 'consensus' in shelf_results:
            shelf_consensus = shelf_results['consensus']

        for position in range(1, 7):  # 1 to 6
            if position in shelf_consensus:
                number = shelf_consensus[position]['number']
                result_parts.append(f"{position}:{number}")
            else:
                result_parts.append(f"{position}:x")  # Missing position marked as x

        # Process area results (positions a-f)
        area_consensus = {}
        if area_results.get('success') and 'consensus' in area_results:
            area_consensus = area_results['consensus']

        for position in ['a', 'b', 'c', 'd', 'e', 'f']:
            if position in area_consensus:
                number = area_consensus[position]['number']
                result_parts.append(f"{position}:{number}")
            else:
                result_parts.append(f"{position}:x")  # Missing position marked as x

        return ",".join(result_parts)

    def process_camera_data(self, camera_data: Dict) -> Dict:
        """Process data provided by camera module"""
        try:
            if not camera_data.get('success'):
                return {
                    'success': False,
                    'error': f"Invalid camera data: {camera_data.get('error', 'Unknown error')}"
                }

            shelf_images = camera_data.get('shelf_images', [])
            area_images = camera_data.get('area_images', [])

            if not shelf_images or not area_images:
                return {
                    'success': False,
                    'error': 'Image data is empty'
                }

            print(f"\nStarting YOLO recognition...")
            print(f"Processing {len(shelf_images)} shelf images and {len(area_images)} area images")
            print(
                f"Shelf inference resolution: {self.shelf_inference_size}, Area inference resolution: {self.area_inference_size}")

            # Recognize shelf
            print("\nShelf Recognition:")
            shelf_results = self.detect_shelf_numbers(shelf_images)

            # Recognize area
            print("\nArea Recognition:")
            area_results = self.detect_area_numbers(area_images)

            # Format result string
            result_string = self.format_result_string(shelf_results, area_results)

            print(f"\nRecognition complete!")
            print(f"Result string: {result_string}")

            return {
                'success': True,
                'shelf_results': shelf_results,
                'area_results': area_results,
                'result_string': result_string,
                'timestamp': camera_data.get('timestamp', ''),
                'processed_at': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'inference_settings': {
                    'shelf_size': self.shelf_inference_size,
                    'area_size': self.area_inference_size
                }
            }

        except Exception as e:
            print(f"Failed to process camera data: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def test_detection(self):
        """Independent test functionality"""
        print("=" * 50)
        print("YOLO Recognition Module Test")
        print("=" * 50)

        # Check model status
        print("\nModel Status Check:")
        print(f"  Device: {self.device}")
        
        # Display shelf model info
        shelf_model_path = self.config["yolo"]["shelf_model_path"]
        shelf_info = self.get_model_info(self.shelf_model, shelf_model_path)
        print(f"  Shelf model: {'Loaded' if self.shelf_model else 'Not loaded'} ({shelf_info})")
        
        # Display area model info  
        area_model_path = self.config["yolo"]["area_model_path"]
        area_info = self.get_model_info(self.area_model, area_model_path)
        print(f"  Area model: {'Loaded' if self.area_model else 'Not loaded'} ({area_info})")
        
        print(f"  Shelf inference resolution: {self.shelf_inference_size}")
        print(f"  Area inference resolution: {self.area_inference_size}")

        if not self.shelf_model or not self.area_model:
            print("\nModels not loaded correctly, please check configuration and model files")
            print("Supported formats: .pt (PyTorch), .onnx (ONNX)")
            return

        print("\nTest complete!")


def main():
    """Main function - test entry point"""
    try:
        import sys

        # Create YOLO module instance
        yolo_module = YOLOModule()

        if len(sys.argv) > 1 and sys.argv[1] == "--with-camera":
            # Integration test with camera module
            try:
                from camera_module import CameraModule

                print("Starting Camera-YOLO integration test...")

                # Create camera module
                camera = CameraModule()

                # Get image data
                camera_data = camera.get_images_for_yolo(count=3, interval_ms=500)

                if camera_data['success']:
                    print(f"Camera acquired {camera_data['count']} images")

                    # YOLO processing
                    result = yolo_module.process_camera_data(camera_data)

                    if result['success']:
                        print(f"\nIntegration test successful!")
                        print(f"Final result string: {result['result_string']}")
                        print(f"Inference settings: {result['inference_settings']}")
                    else:
                        print(f"YOLO processing failed: {result.get('error')}")
                else:
                    print(f"Camera data acquisition failed: {camera_data.get('error')}")

                # Clean up camera resources
                camera.release_cameras()

            except ImportError:
                print("Failed to import CameraModule, ensure camera_module.py is in the same directory")
            except Exception as e:
                print(f"Integration test failed: {e}")
        else:
            # Independent test
            yolo_module.test_detection()

    except Exception as e:
        print(f"Program runtime error: {e}")


if __name__ == "__main__":
    main()