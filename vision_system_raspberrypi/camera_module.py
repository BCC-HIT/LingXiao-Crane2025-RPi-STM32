#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera Module - Responsible for USB camera initialization, configuration and image capture
Supports automatic camera detection, batch capture, independent testing and other functions
"""

import cv2
import os
import time
import json
import subprocess
from typing import List, Tuple, Optional, Dict
from datetime import datetime


class CameraModule:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize camera module

        Args:
            config_path: Configuration file path
        """
        # Reduce OpenCV warning messages (compatible with old OpenCV versions)
        try:
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
        except AttributeError:
            # Old OpenCV versions don't have LOG_LEVEL_ERROR, skip
            pass

        self.config = self.load_config(config_path)
        self.shelf_camera = None  # Shelf camera
        self.area_camera = None  # Area camera
        self.shelf_camera_id = None
        self.area_camera_id = None
        self.shelf_camera_device_path = None
        self.area_camera_device_path = None
        self.available_cameras = []  # Cache detected camera list
        self.single_camera_mode = False  # Single camera mode flag

        # Create test image save directory
        os.makedirs("test_images", exist_ok=True)

    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        default_config = {
            "camera": {
                "shelf_camera_device": "/dev/camera_shelf",
                "area_camera_device": "/dev/camera_area",
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "capture_count": 5,
                "default_capture_interval_ms": 200,
                # Automatic control switches
                "auto_exposure": True,
                "auto_white_balance": True,
                # Basic image parameters
                "brightness": 0,  # Brightness (-64 to 64)
                "contrast": 32,  # Contrast (0 to 64)
                "saturation": 64,  # Saturation (0 to 128)
                "hue": 0,  # Hue (-40 to 40)
                "gamma": 100,  # Gamma value (72 to 500)
                "gain": 0,  # Gain (0 to 100)
                "sharpness": 2,  # Sharpness (0 to 6)
                # Advanced parameters
                "power_line_frequency": 1,  # Power frequency (0=Disabled, 1=50Hz, 2=60Hz)
                "white_balance_temperature": 4600,  # White balance temperature (2800 to 6500)
                "backlight_compensation": 1,  # Backlight compensation (0 to 2)
                "exposure_time_absolute": 157,  # Absolute exposure time (1 to 5000)
                "exposure_dynamic_framerate": True,  # Dynamic framerate (default enabled)
                # System settings
                "use_v4l2_controls": True
            }
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
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

    def detect_cameras(self, force_detect: bool = False) -> List[int]:
        """
        Detect available cameras

        Args:
            force_detect: Whether to force re-detection

        Returns:
            Available camera ID list
        """
        # If already detected and not forcing re-detection, return cached results directly
        if self.available_cameras and not force_detect:
            return self.available_cameras

        available_cameras = []
        print("Detecting cameras...")

        # First check /dev/video* devices
        video_devices = []
        for i in range(10):
            if os.path.exists(f'/dev/video{i}'):
                video_devices.append(i)

        if video_devices:
            print(f"Found video devices: {video_devices}")

        # Detect available cameras (reduce unnecessary attempts)
        test_indices = video_devices if video_devices else range(4)

        for i in test_indices:
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)  # Explicitly use V4L2 backend
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        available_cameras.append(i)
                        # Get camera information
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = int(cap.get(cv2.CAP_PROP_FPS))
                        print(f"Camera {i}: {width}x{height}@{fps}fps")
                cap.release()
            except Exception as e:
                # Silently handle exceptions, reduce output
                pass

        # Cache detection results
        self.available_cameras = available_cameras

        if len(available_cameras) == 0:
            print("No cameras detected!")
        elif len(available_cameras) == 1:
            print(f"Only 1 camera detected: {available_cameras[0]} (Single camera mode will be enabled)")
        else:
            print(f"Detected {len(available_cameras)} cameras: {available_cameras}")

        return available_cameras

    def set_v4l2_controls(self, device_path: str, camera_name: str) -> bool:
        """
        Set camera parameters using v4l2-ctl

        Args:
            device_path: Camera device path
            camera_name: Camera name

        Returns:
            Whether settings were successful
        """
        if not self.config["camera"].get("use_v4l2_controls", False):
            return True

        try:
            camera_config = self.config["camera"]
            device = device_path

            # Check if device exists
            if not os.path.exists(device):
                print(f"Device {device} does not exist")
                return False

            controls = []

            # Auto exposure settings (3=Auto, 1=Manual)
            if camera_config.get("auto_exposure", True):
                controls.append("auto_exposure=3")
            else:
                controls.append("auto_exposure=1")
                # Set exposure time in manual mode
                if "exposure_time_absolute" in camera_config:
                    controls.append(f"exposure_time_absolute={camera_config['exposure_time_absolute']}")

            # Auto white balance
            if camera_config.get("auto_white_balance", True):
                controls.append("white_balance_automatic=1")
            else:
                controls.append("white_balance_automatic=0")
                # Set white balance temperature in manual mode
                if "white_balance_temperature" in camera_config:
                    controls.append(f"white_balance_temperature={camera_config['white_balance_temperature']}")

            # Basic image parameters
            basic_params = ["brightness", "contrast", "saturation", "hue", "gamma", "gain", "sharpness"]
            for param in basic_params:
                if param in camera_config:
                    controls.append(f"{param}={camera_config[param]}")

            # Advanced parameters
            advanced_params = ["power_line_frequency", "backlight_compensation"]
            for param in advanced_params:
                if param in camera_config:
                    controls.append(f"{param}={camera_config[param]}")

            # Dynamic framerate settings
            if "exposure_dynamic_framerate" in camera_config:
                value = 1 if camera_config["exposure_dynamic_framerate"] else 0
                controls.append(f"exposure_dynamic_framerate={value}")

            # Execute v4l2-ctl command
            if controls:
                cmd = ["v4l2-ctl", "-d", device, "--set-ctrl"] + [",".join(controls)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

                if result.returncode == 0:
                    print(f"{camera_name} v4l2 parameters set successfully")
                    return True
                else:
                    print(f"{camera_name} v4l2 parameter setting failed: {result.stderr}")
                    return False

            return True

        except subprocess.TimeoutExpired:
            print(f"{camera_name} v4l2 setting timed out")
            return False
        except FileNotFoundError:
            print("v4l2-ctl command not found, please install v4l-utils")
            return True  # Don't block program execution
        except Exception as e:
            print(f"{camera_name} Exception setting v4l2 parameters: {e}")
            return False

    def configure_camera(self, cap: cv2.VideoCapture, camera_name: str, device_path: str) -> bool:
        """
        Configure camera parameters

        Args:
            cap: OpenCV camera object
            camera_name: Camera name (for logging)
            device_path: Camera device path

        Returns:
            Whether configuration was successful
        """
        try:
            camera_config = self.config["camera"]

            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])

            # Set framerate
            cap.set(cv2.CAP_PROP_FPS, camera_config["fps"])

            # OpenCV auto exposure settings (backup solution)
            if camera_config.get("auto_exposure", True):
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure

            # OpenCV auto white balance settings
            if camera_config.get("auto_white_balance", True):
                cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance

            # Set buffer size (reduce latency)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Use v4l2-ctl to set more precise parameters
            v4l2_success = self.set_v4l2_controls(device_path, camera_name)

            # Verify settings
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

            status = "OK" if v4l2_success else "WARN"
            print(f"{camera_name} configuration complete {status}: {actual_width}x{actual_height}@{actual_fps}fps")
            return True

        except Exception as e:
            print(f"{camera_name} configuration failed: {e}")
            return False

    def init_cameras(self) -> bool:
        """
        Initialize two cameras (or single camera mode)

        Returns:
            Whether initialization was successful
        """
        try:
            camera_config = self.config["camera"]
            shelf_device_path = camera_config.get("shelf_camera_device")
            area_device_path = camera_config.get("area_camera_device")

            if not shelf_device_path or not area_device_path:
                print("Error: shelf_camera_device or area_camera_device not specified in configuration file")
                return False

            self.shelf_camera_device_path = shelf_device_path
            self.area_camera_device_path = area_device_path

            print(f"Configured shelf camera device: {self.shelf_camera_device_path}")
            print(f"Configured area camera device: {self.area_camera_device_path}")

            # Check if device paths exist (udev rules might not be effective yet or misconfigured)
            if not os.path.exists(self.shelf_camera_device_path):
                print(f"Error: Shelf camera device path '{self.shelf_camera_device_path}' does not exist!")
                self.detect_cameras(force_detect=True)  # Print available /dev/videoX for debugging
                return False

            # Determine single/dual camera mode
            if self.shelf_camera_device_path == self.area_camera_device_path:
                self.single_camera_mode = True
                print(f"\nEnabling single camera mode (Device: {self.shelf_camera_device_path})")
                print("The same camera will be used for both shelf and area views")

                print(f"\nInitializing camera (Device: {self.shelf_camera_device_path})...")
                self.shelf_camera = cv2.VideoCapture(self.shelf_camera_device_path, cv2.CAP_V4L2)
                if not self.shelf_camera.isOpened():
                    print(f"Camera {self.shelf_camera_device_path} initialization failed!")
                    return False

                if not self.configure_camera(self.shelf_camera, "Camera", self.shelf_camera_device_path):
                    return False

                self.area_camera = self.shelf_camera  # Point to the same object
                print(f"\nSingle camera mode initialized successfully!")
                return True
            else:
                # Dual camera mode
                self.single_camera_mode = False
                if not os.path.exists(self.area_camera_device_path):
                    print(f"Error: Area camera device path '{self.area_camera_device_path}' does not exist!")
                    self.detect_cameras(force_detect=True)  # Print available /dev/videoX for debugging
                    return False

                # Initialize shelf camera
                print(f"\nInitializing shelf camera (Device: {self.shelf_camera_device_path})...")
                self.shelf_camera = cv2.VideoCapture(self.shelf_camera_device_path, cv2.CAP_V4L2)
                if not self.shelf_camera.isOpened():
                    print(f"Shelf camera {self.shelf_camera_device_path} initialization failed!")
                    return False
                if not self.configure_camera(self.shelf_camera, "Shelf Camera", self.shelf_camera_device_path):
                    return False

                # Initialize area camera
                print(f"\nInitializing area camera (Device: {self.area_camera_device_path})...")
                self.area_camera = cv2.VideoCapture(self.area_camera_device_path, cv2.CAP_V4L2)
                if not self.area_camera.isOpened():
                    print(f"Area camera {self.area_camera_device_path} initialization failed!")
                    return False
                if not self.configure_camera(self.area_camera, "Area Camera", self.area_camera_device_path):
                    return False

                print("\nDual camera mode initialized successfully!")
                return True

        except Exception as e:
            print(f"Camera initialization failed: {e}")
            return False

    def capture_single_image(self, camera: cv2.VideoCapture, camera_name: str) -> Optional[any]:
        """
        Single image capture

        Args:
            camera: Camera object
            camera_name: Camera name

        Returns:
            Captured image, returns None on failure
        """
        try:
            # Clear buffer to get latest frame
            for _ in range(1):
                ret, frame = camera.read()
                if not ret:
                    print(f"{camera_name} read failed")
                    return None

            return frame

        except Exception as e:
            print(f"{camera_name} capture failed: {e}")
            return None

    def capture_images(self, count: int = None) -> Tuple[List, List]:
        """
        Batch image capture

        Args:
            count: Number of captures, uses configuration file count when None

        Returns:
            (shelf image list, area image list)
        """
        if count is None:
            count = self.config["camera"]["capture_count"]

        if not self.shelf_camera or not self.area_camera:
            print("Cameras not initialized!")
            return [], []

        shelf_images = []
        area_images = []

        print(f"\nStarting batch capture, {count} images per camera...")

        if self.single_camera_mode:
            print("Single camera mode: Using the same camera for shelf and area views")

        for i in range(count):
            print(f"Capturing image {i + 1}/{count}...")

            if self.single_camera_mode:
                # Single camera mode: use same camera to capture twice
                shelf_img = self.capture_single_image(self.shelf_camera, "Camera(Shelf Angle)")
                # Brief wait for camera stability
                time.sleep(0.1)
                area_img = self.capture_single_image(self.shelf_camera, "Camera(Area Angle)")
            else:
                # Dual camera mode: capture both cameras simultaneously
                shelf_img = self.capture_single_image(self.shelf_camera, "Shelf Camera")
                area_img = self.capture_single_image(self.area_camera, "Area Camera")

            if shelf_img is not None:
                shelf_images.append(shelf_img)
            if area_img is not None:
                area_images.append(area_img)

            # Brief interval
            time.sleep(0.1)

        print(f"Capture complete: {len(shelf_images)} shelf images, {len(area_images)} area images")
        return shelf_images, area_images

    def capture_images_with_interval(self, count: int = 5, interval_ms: Optional[int] = None) -> Tuple[List, List]:
        """
        Capture specified number of photos at specified intervals

        Args:
            count: Number of captures
            interval_ms: Capture interval (milliseconds)

        Returns:
            (shelf image list, area image list)
        """
        if not self.shelf_camera or not self.area_camera:
            print("Cameras not initialized!")
            return [], []

        shelf_images = []
        area_images = []

        # Determine actual interval time to use
        actual_interval_ms = interval_ms
        if actual_interval_ms is None:  # If interval_ms not provided when called
            actual_interval_ms = self.config["camera"].get("default_capture_interval_ms", 200)

        print(f"\nStarting interval capture: {count} images, {actual_interval_ms}ms interval...")

        if self.single_camera_mode:
            print("Single camera mode: Using the same camera for shelf and area views")

        for i in range(count):
            print(f"Capturing image {i + 1}/{count}...")

            if self.single_camera_mode:
                # Single camera mode: use same camera to capture twice
                shelf_img = self.capture_single_image(self.shelf_camera, "Camera(Shelf Angle)")
                # Brief wait for camera stability
                time.sleep(0.1)
                area_img = self.capture_single_image(self.shelf_camera, "Camera(Area Angle)")
            else:
                # Dual camera mode: capture both cameras simultaneously
                shelf_img = self.capture_single_image(self.shelf_camera, "Shelf Camera")
                area_img = self.capture_single_image(self.area_camera, "Area Camera")

            if shelf_img is not None:
                shelf_images.append(shelf_img)
            if area_img is not None:
                area_images.append(area_img)

            # If not the last one, wait specified interval
            if i < count - 1:
                time.sleep(actual_interval_ms / 1000.0)

        print(f"Interval capture complete: {len(shelf_images)} shelf images, {len(area_images)} area images")
        return shelf_images, area_images

    def get_images_for_yolo(self, count: Optional[int] = None, interval_ms: Optional[int] = None) -> Dict:
        """
        Get image data for YOLO recognition
        """
        try:
            # Ensure cameras are initialized
            if not self.shelf_camera or not self.area_camera:
                if not self.init_cameras():
                    return {
                        'shelf_images': [], 'area_images': [],
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                        'success': False, 'error': 'Camera initialization failed'
                    }

            # Determine capture count and interval, read from configuration if not provided
            actual_count = count
            if actual_count is None:
                actual_count = self.config["camera"].get("capture_count", 5)

            actual_interval_ms = interval_ms
            if actual_interval_ms is None:
                actual_interval_ms = self.config["camera"].get("default_capture_interval_ms", 200)

            # Call capture function only once
            shelf_images, area_images = self.capture_images_with_interval(actual_count, actual_interval_ms)

            return {
                'shelf_images': shelf_images,
                'area_images': area_images,
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'success': len(shelf_images) > 0 and len(area_images) > 0,
                'count': len(shelf_images)
            }

        except Exception as e:
            return {
                'shelf_images': [], 'area_images': [],
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'success': False, 'error': str(e)
            }

    def save_images(self, shelf_images: List, area_images: List, prefix: str = "test") -> None:
        """
        Save images to file

        Args:
            shelf_images: Shelf image list
            area_images: Area image list
            prefix: Filename prefix
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save shelf images
        for i, img in enumerate(shelf_images):
            filename = f"test_images/{prefix}_shelf_{timestamp}_{i + 1}.jpg"
            cv2.imwrite(filename, img)
            print(f"Saved shelf image: {filename}")

        # Save area images
        for i, img in enumerate(area_images):
            filename = f"test_images/{prefix}_area_{timestamp}_{i + 1}.jpg"
            cv2.imwrite(filename, img)
            print(f"Saved area image: {filename}")

    def release_cameras(self) -> None:
        """Release camera resources"""
        if self.single_camera_mode:
            # Single camera mode: release only once
            if self.shelf_camera:
                self.shelf_camera.release()
                self.shelf_camera = None
                self.area_camera = None  # Points to same object, set to None
        else:
            # Dual camera mode: release separately
            if self.shelf_camera:
                self.shelf_camera.release()
                self.shelf_camera = None

            if self.area_camera:
                self.area_camera.release()
                self.area_camera = None

    def show_camera_info(self, device_path: str, camera_name: str) -> None:
        """
        Display detailed camera information

        Args:
            device_path: Camera device path
            camera_name: Camera name
        """
        try:
            device = device_path
            if not os.path.exists(device):
                print(f"  Device {device} does not exist, cannot display information.")
                return

            print(f"\n{camera_name} ({device}) Detailed Parameters:")
            print("-" * 60)

            # Display all available control parameters
            cmd = ["v4l2-ctl", "-d", device, "--list-ctrls"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                current_category = None
                for line in result.stdout.strip().split('\n'):
                    line = line.strip()
                    if not line:
                        continue
                    if line.endswith("Controls"):  # User Controls, Camera Controls etc.
                        current_category = line
                        print(f"\n {current_category}:")
                    elif ":" in line and "0x" in line:  # Parameter line
                        # Parse parameter line, extract key information
                        parts = line.split(":")
                        if len(parts) >= 2:
                            param_name = parts[0].strip()
                            param_info = parts[1].strip()
                            print(f"  - {param_name}: {param_info}")
            else:
                print("  Could not get parameter information")

            print("-" * 60)

            # Display key automatic control states and exposure-related parameters
            print(f"\n{camera_name} Key Control States:")

            # Get auto exposure state
            cmd = ["v4l2-ctl", "-d", device, "--get-ctrl", "auto_exposure"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                exposure_line = result.stdout.strip()
                print(f"  {exposure_line}")
            else:
                print("  auto_exposure: failed to get")

            # Get absolute exposure time
            cmd = ["v4l2-ctl", "-d", device, "--get-ctrl", "exposure_time_absolute"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                exposure_time_line = result.stdout.strip()
                print(f"  {exposure_time_line}")
            else:
                print("  exposure_time_absolute: failed to get")

            # Get dynamic framerate state
            cmd = ["v4l2-ctl", "-d", device, "--get-ctrl", "exposure_dynamic_framerate"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                dynamic_framerate_line = result.stdout.strip()
                print(f"  {dynamic_framerate_line}")
            else:
                print("  exposure_dynamic_framerate: failed to get")

            # Get auto white balance state
            cmd = ["v4l2-ctl", "-d", device, "--get-ctrl", "white_balance_automatic"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                wb_line = result.stdout.strip()
                print(f"  {wb_line}")
            else:
                print("  white_balance_automatic: failed to get")

        except subprocess.TimeoutExpired:
            print(f"  Timeout getting {camera_name} information")
        except FileNotFoundError:
            print("  v4l2-ctl not installed. Please run: sudo apt-get install v4l-utils")
        except Exception as e:
            print(f"  Failed to get {camera_name} information: {e}")

    def list_camera_capabilities(self, device_path: str) -> None:
        """
        List camera supported formats and resolutions

        Args:
            device_path: Camera device path
        """
        try:
            device = device_path
            if not os.path.exists(device):
                print(f"  Device {device} does not exist, cannot list capabilities.")
                return

            print(f"\nCamera {device} Supported Formats:")
            print("-" * 50)

            cmd = ["v4l2-ctl", "-d", device, "--list-formats-ext"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                current_format = None
                for line in lines:
                    line = line.strip()
                    if line.startswith("["):  # Format line
                        current_format = line
                        print(f"\n {current_format}")
                    elif "Size:" in line and "fps" in line:  # Resolution and framerate line
                        print(f"  - {line}")
            else:
                print("  Could not get format information")

        except Exception as e:
            print(f"  Failed to get format information: {e}")

    def test_single_method(self, method_name: str) -> None:
        """
        Test single method

        Args:
            method_name: Method name
        """
        print(f"\nTesting method: {method_name}")
        print("=" * 40)

        if method_name == "detect_cameras":
            cameras = self.detect_cameras(force_detect=True)  # Force re-detection
            print(f"Detection result: {cameras}")
            if len(cameras) == 1:
                print("  Single camera detected, single camera mode will be enabled")
            elif len(cameras) >= 2:
                print("  Multiple cameras detected, dual camera mode will be enabled")

        elif method_name == "init_cameras":
            success = self.init_cameras()
            print(f"Initialization result: {'Success' if success else 'Failure'}")

        elif method_name == "capture_single":
            if not self.shelf_camera or not self.area_camera:
                print("Cameras not initialized, auto-initializing...")
                if not self.init_cameras():
                    print("Camera initialization failed")
                    return
            print("Capturing single image...")
            shelf_img = self.capture_single_image(self.shelf_camera, "Shelf Camera")
            area_img = self.capture_single_image(self.area_camera, "Area Camera")
            if shelf_img is not None and area_img is not None:
                self.save_images([shelf_img], [area_img], "method_test")
                print("Single capture successful")
            else:
                print("Single capture failed")

        elif method_name == "capture_batch":
            if not self.shelf_camera or not self.area_camera:
                print("Cameras not initialized, auto-initializing...")
                if not self.init_cameras():
                    print("Camera initialization failed")
                    return
            print("Capturing batch images...")
            shelf_images, area_images = self.capture_images(3)  # Test with 3 images
            if shelf_images and area_images:
                self.save_images(shelf_images, area_images, "batch_test")
                print("Batch capture successful")
            else:
                print("Batch capture failed")

        elif method_name == "capture_interval":
            if not self.shelf_camera or not self.area_camera:
                print("Cameras not initialized, auto-initializing...")
                if not self.init_cameras():
                    print("Camera initialization failed")
                    return
            print("Interval capture test (3 images, 500ms interval)...")
            shelf_images, area_images = self.capture_images_with_interval(3, 500)
            if shelf_images and area_images:
                self.save_images(shelf_images, area_images, "interval_test")
                print("Interval capture successful")
            else:
                print("Interval capture failed")

        elif method_name == "yolo_ready":
            print("Testing YOLO data acquisition...")
            data = self.get_images_for_yolo(3, 300)
            if data['success']:
                print(f"YOLO data preparation successful: {data['count']} images")
                print(f"  Timestamp: {data['timestamp']}")
                # Optional: save test images
                self.save_images(data['shelf_images'], data['area_images'], "yolo_ready")
            else:
                print(f"YOLO data preparation failed: {data.get('error', 'Unknown error')}")

        elif method_name == "show_info":
            # Prioritize displaying configured camera information
            if self.config["camera"].get("shelf_camera_device"):
                self.show_camera_info(self.config["camera"]["shelf_camera_device"], "Configured Shelf Camera")
            if self.config["camera"].get("area_camera_device") and \
                    self.config["camera"]["area_camera_device"] != self.config["camera"].get("shelf_camera_device"):
                self.show_camera_info(self.config["camera"]["area_camera_device"], "Configured Area Camera")
            else:  # If not in configuration, can try to list all detected /dev/videoX
                print(
                    "Camera device paths not fully specified in config, attempting to list all detected /dev/videoX...")
                available_cameras = self.detect_cameras()  # Still can be used for enumeration
                for cam_id in available_cameras[:2]:
                    self.show_camera_info(f"/dev/video{cam_id}", f"Detected Camera /dev/video{cam_id}")

        elif method_name == "show_formats":
            # Similarly adjust
            if self.config["camera"].get("shelf_camera_device"):
                self.list_camera_capabilities(self.config["camera"]["shelf_camera_device"])
            if self.config["camera"].get("area_camera_device") and \
                    self.config["camera"]["area_camera_device"] != self.config["camera"].get("shelf_camera_device"):
                self.list_camera_capabilities(self.config["camera"]["area_camera_device"])
            else:
                print(
                    "Camera device paths not fully specified in config, attempting to list all detected /dev/videoX...")
                available_cameras = self.detect_cameras()
                for cam_id in available_cameras[:2]:
                    self.list_camera_capabilities(f"/dev/video{cam_id}")

        else:
            print(f"Unknown method: {method_name}")
            print(
                "Available methods: detect_cameras, init_cameras, capture_single, capture_batch, capture_interval, yolo_ready, show_info, show_formats")

    def print_all_methods(self) -> None:
        """Print all available methods and descriptions"""
        print("\nCameraModule All Methods Description:")
        print("=" * 60)

        methods = [
            ("__init__(config_path)", "Initializes the camera module, loads configuration"),
            ("load_config(config_path)", "Loads the JSON configuration file"),
            ("detect_cameras()", "Automatically detects available USB cameras"),
            ("set_v4l2_controls(camera_id, name)", "Sets camera parameters using v4l2-ctl"),
            ("configure_camera(cap, name, id)", "Configures an OpenCV camera object"),
            ("init_cameras()", "Initializes the shelf and area cameras"),
            ("capture_single_image(camera, name)", "Captures a single image"),
            ("capture_images(count)", "Captures a batch of images, returns a list of images"),
            ("capture_images_with_interval(count, interval_ms)",
             "Captures a specified number of photos at a given interval"),
            ("get_images_for_yolo(count, interval_ms)", "Prepares image data for YOLO recognition"),
            ("save_images(shelf_imgs, area_imgs, prefix)", "Saves images to a file"),
            ("release_cameras()", "Releases camera resources"),
            ("show_camera_info(camera_id, name)", "Displays detailed camera parameter information"),
            ("list_camera_capabilities(camera_id)", "Lists supported formats and resolutions of a camera"),
            ("test_cameras()", "Full interactive test"),
            ("test_single_method(method_name)", "Tests a single method"),
        ]

        for i, (method, desc) in enumerate(methods, 1):
            print(f"{i:2d}. {method:<35} - {desc}")

        print("\nUsage examples:")
        print("python camera_module.py                           # Full test")
        print("python camera_module.py --test-method detect_cameras    # Test a single method")
        print("python camera_module.py --test-method capture_interval  # Test interval capture")
        print("python camera_module.py --test-method yolo_ready        # Test YOLO data preparation")
        print("python camera_module.py --list-methods           # Display all methods")

    def test_cameras(self) -> None:
        """
        Camera test functionality
        Includes real-time preview and capture testing
        """
        print("=" * 50)
        print("Camera Module Test")
        print("=" * 50)

        # 1. Detect cameras (only detect once)
        available_cameras = self.detect_cameras()
        if len(available_cameras) == 0:
            print("Test failed: No cameras detected")
            return

        # 2. Initialize cameras (won't duplicate detection)
        if not self.init_cameras():
            print("Test failed: Camera initialization failed")
            return

        # 3. Display detailed camera information
        if self.single_camera_mode:
            if self.shelf_camera_device_path:  # Use correct attribute name
                self.show_camera_info(self.shelf_camera_device_path, f"Camera ({self.shelf_camera_device_path})")
        else:
            if self.shelf_camera_device_path:
                self.show_camera_info(self.shelf_camera_device_path,
                                      f"Shelf Camera ({self.shelf_camera_device_path})")
            if self.area_camera_device_path:
                self.show_camera_info(self.area_camera_device_path,
                                      f"Area Camera ({self.area_camera_device_path})")

        print("\nTest Instructions:")
        print("- Press SPACE to capture and save an image")
        print("- Press 's' for batch capture")
        print("- Press ESC to exit the test")

        if self.single_camera_mode:
            print("\nSingle camera mode: Shelf and area will use the same camera")

        print("\nDisplaying live preview...")

        try:
            while True:
                # Read camera frames
                if self.single_camera_mode:
                    # Single camera mode: display one window
                    ret, frame = self.shelf_camera.read()
                    if ret:
                        # Scale for display (1920x1080 too large, scale to 640x360 for display)
                        display_frame = cv2.resize(frame, (640, 360))

                        # Add identifier to image
                        cv2.putText(display_frame, f"Camera (Path: {self.shelf_camera_device_path}) - Single Mode",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        # Display frame
                        cv2.imshow("Camera Preview", display_frame)
                else:
                    # Dual camera mode: display two windows
                    ret1, frame1 = self.shelf_camera.read()
                    ret2, frame2 = self.area_camera.read()

                    if ret1 and ret2:
                        # Scale for display (1920x1080 too large, scale to 640x360 for display)
                        display_frame1 = cv2.resize(frame1, (640, 360))
                        display_frame2 = cv2.resize(frame2, (640, 360))

                        # Add identifier to images
                        cv2.putText(display_frame1, f"Shelf Camera (Path: {self.shelf_camera_device_path})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(display_frame2, f"Area Camera (Path: {self.area_camera_device_path})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Display frames
                        cv2.imshow("Shelf Camera", display_frame1)
                        cv2.imshow("Area Camera", display_frame2)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key to exit
                    break
                elif key == ord(' '):  # Space key to capture
                    print("\nSingle capture test...")
                    if self.single_camera_mode:
                        # Single camera mode: capture same camera twice
                        shelf_img = self.capture_single_image(self.shelf_camera, "Camera(Shelf Angle)")
                        area_img = self.capture_single_image(self.shelf_camera, "Camera(Area Angle)")
                    else:
                        shelf_img = self.capture_single_image(self.shelf_camera, "Shelf Camera")
                        area_img = self.capture_single_image(self.area_camera, "Area Camera")

                    if shelf_img is not None and area_img is not None:
                        self.save_images([shelf_img], [area_img], "single")
                        print("Single capture complete!")
                    else:
                        print("Single capture failed!")

                elif key == ord('s'):  # 's' key for batch capture
                    print("\nBatch capture test...")
                    shelf_images, area_images = self.capture_images()
                    if shelf_images and area_images:
                        self.save_images(shelf_images, area_images, "batch")
                        print("Batch capture complete!")
                    else:
                        print("Batch capture failed!")

        except KeyboardInterrupt:
            print("\nUser interrupted test")
        except Exception as e:
            print(f"\nAn error occurred during the test: {e}")
        finally:
            # Clean up resources
            cv2.destroyAllWindows()
            self.release_cameras()
            print("\nTest finished, resources cleaned up")


def main():
    """Main function - independent test entry point"""
    try:
        # Check command line parameters
        import sys
        if len(sys.argv) > 1:
            mode = sys.argv[1]
            camera_module = CameraModule()

            if mode == "--test-method":
                if len(sys.argv) > 2:
                    method_name = sys.argv[2]
                    camera_module.test_single_method(method_name)
                else:
                    print("Usage: python camera_module.py --test-method <method_name>")
                    print(
                        "Available methods: detect_cameras, init_cameras, capture_single, capture_batch, capture_interval, yolo_ready, show_info, show_formats")
                return
            elif mode == "--list-methods":
                camera_module.print_all_methods()
                return

        # Create camera module instance
        camera_module = CameraModule()

        # Run complete test
        camera_module.test_cameras()

    except Exception as e:
        print(f"Program runtime error: {e}")
    finally:
        # Ensure resource cleanup
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()