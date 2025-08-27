#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Intelligent Delivery Robot Vision Recognition System - Main Program
New Features:
1. Signal-controlled detection flow (start/pause/resume/stop)
2. Configurable YOLO inference resolution
3. Improved position mapping algorithm
4. Enhanced error handling and state management
5. Fixed signal control issues and added debug mode
6. Complete PERFECT result functionality - saves and sends perfect recognition results
"""

import time

print(f"--- Python script started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---")
import json
import os
import logging
import signal
import sys
import cv2
import numpy as np
import threading
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from camera_module import CameraModule
from yolo_module import YOLOModule
from uart_module import UARTModule, SystemState
from collections import deque


class ContinuousMatchingTracker:
    """Continuous matching tracker (fixed version)"""

    def __init__(self, required_matches: int = 3):
        self.required_matches = required_matches
        self.history = deque(maxlen=required_matches)
        self.match_count = 0
        self.last_result = None
        self.is_confirmed = False

    def add_result(self, result_string: str) -> Dict:
        self.history.append(result_string)

        if result_string == self.last_result:
            self.match_count += 1
        else:
            self.match_count = 1
            self.last_result = result_string
            self.is_confirmed = False

        is_matched = self.match_count >= self.required_matches
        is_new_confirmation = is_matched and not self.is_confirmed
        if is_new_confirmation:
            self.is_confirmed = True

        return {
            'matched': is_matched,
            'is_new_confirmation': is_new_confirmation,
            'current_count': self.match_count,
            'required_count': self.required_matches,
            'result': result_string if is_matched else None,
            'history': list(self.history)
        }

    def reset(self):
        self.history.clear()
        self.match_count = 0
        self.last_result = None
        self.is_confirmed = False

    def get_status(self) -> Dict:
        return {
            'current_count': self.match_count,
            'required_count': self.required_matches,
            'last_result': self.last_result,
            'is_confirmed': self.is_confirmed,
            'history': list(self.history)
        }


class VisionSystemV2:
    """Intelligent Delivery Robot Vision Recognition System Main Class - V2"""

    def __init__(self, config_path: str = "config.json"):
        """
        Initialize vision recognition system

        Args:
            config_path: Configuration file path
        """
        self.config_path = config_path
        self.config = self.load_config()

        # Initialize modules
        self.camera = None
        self.yolo = None
        self.uart = None
        self.running = False

        # Signal control related
        self.enable_signal_control = self.config['uart'].get('enable_signal_control', False)
        self.enable_pause_control = self.config['uart'].get('enable_pause_control', False)
        self.wait_for_start_signal = self.config['system'].get('wait_for_start_signal', False)

        # Visualization related
        self.display_active = False
        self.current_shelf_image = None
        self.current_area_image = None
        self.current_detections = {'shelf': [], 'area': []}
        self.current_result_string = ""
        self.recognition_thread = None

        # System statistics
        self.stats = {
            'total_runs': 0,
            'successful_runs': 0,
            'failed_runs': 0,
            'paused_runs': 0,
            'start_time': None,
            'last_result': '',
            'last_run_time': None,
            'signals_processed': 0,
            'perfect_results_found': 0  # Track number of perfect results found
        }

        # Setup logging
        self.setup_logging()

        # Register signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        print(f"Vision System initialization complete")
        print(f"Signal Control: {'Enabled' if self.enable_signal_control else 'Disabled'}, "
              f"Pause Control: {'Enabled' if self.enable_pause_control else 'Disabled'}, "
              f"Wait for Start Signal: {'Yes' if self.wait_for_start_signal else 'No'}")

        self.last_successful_result = None  # Store last successful result
        self.last_perfect_result = None  # Store last PERFECT result
        self.result_lock = threading.Lock()  # Thread lock for thread-safe access
        self.preview_fps = self.config.get('visual', {}).get('preview_fps', 30)

    def load_config(self) -> Dict:
        """Load system configuration"""
        default_config = {
            "camera": {
                "shelf_camera_device": "/dev/camera_shelf",
                "area_camera_device": "/dev/camera_area",
                "width": 1920,
                "height": 1080,
                "fps": 30,
                "capture_count": 2,
                "default_capture_interval_ms": 50,
                "auto_exposure": True,
                "auto_white_balance": True,
                "brightness": 0,
                "contrast": 32,
                "saturation": 64,
                "hue": 0,
                "gamma": 100,
                "gain": 0,
                "sharpness": 2,
                "power_line_frequency": 1,
                "white_balance_temperature": 4600,
                "backlight_compensation": 1,
                "exposure_time_absolute": 157,
                "exposure_dynamic_framerate": True,
                "use_v4l2_controls": True
            },
            "yolo": {
                "shelf_model_path": "/home/pi/code/yolov11s_1080_shelf.pt",
                "area_model_path": "/home/pi/code/yolov11s_1080_area.pt",
                "shelf_confidence_threshold": 0.763,
                "area_confidence_threshold": 0.615,
                "nms_threshold": 0.4,
                "consensus_method": "voting",
                "min_consensus_count": 2,
                "continuous_matching": {
                    "enabled": True,
                    "required_matches": 2,
                    "check_interval": 0.2,
                    "send_every_match": True
                },
                "device": "cpu",
                "shelf_inference_size": 640,
                "area_inference_size": 640,
                "shelf_position_mapping": {
                    "y_upper_lower_split": 0.49,
                    "x_left_split": 0.33,
                    "x_right_split": 0.67
                },
                "area_position_mapping": {
                    "x_splits": [0.2, 0.35, 0.5, 0.65, 0.8]
                }
            },
            "uart": {
                "port": "/dev/ttyAMA0",
                "baudrate": 9600,
                "stopbits": 2,
                "timeout": 0.5,
                "retry_count": 3,
                "retry_delay": 0.2,
                "message_end": "\n",
                "enable_signal_control": True,
                "enable_pause_control": True,
                "start_signal": "START",
                "pause_signal": "PAUSE",
                "resume_signal": "RESUME",
                "stop_signal": "STOP",
                "request_last_result_signal": "REQUEST",
                "perfect_result_signal": "PERFECT",
                "signal_check_interval": 0.1
            },
            "system": {
                "auto_mode": True,
                "run_interval": 0.3,
                "max_retries": 3,
                "save_debug_images": False,
                "log_level": "INFO",
                "result_send_count": 2,
                "result_send_interval": 0.1,
                "wait_for_start_signal": False,
                "auto_start_timeout": 30,
                "pause_between_cycles": False
            },
            "visual": {
                "display_width": 960,
                "display_height": 540,
                "font_scale": 0.8,
                "line_thickness": 2,
                "text_color": [0, 255, 0],
                "bbox_color": [255, 0, 0],
                "confidence_threshold_display": 0.3,
                "preview_fps": 3
            }
        }

        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Merge default configuration
                    for section, defaults in default_config.items():
                        if section not in config:
                            config[section] = defaults
                        else:
                            for key, value in defaults.items():
                                if key not in config[section]:
                                    config[section][key] = value
                    return config
            else:
                # Create default configuration file
                with open(self.config_path, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=4, ensure_ascii=False)
                print(f"Default configuration file created: {self.config_path}")
                return default_config
        except Exception as e:
            print(f"Configuration loading failed: {e}")
            return default_config

    def setup_logging(self):
        """Setup logging system"""
        log_level = getattr(logging, self.config['system'].get('log_level', 'INFO'))

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Create file handler
        file_handler = logging.FileHandler('vision_system.log', encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        console_handler.setLevel(log_level)

        # Configure logger
        logging.basicConfig(
            level=log_level,
            handlers=[file_handler, console_handler]
        )

        self.logger = logging.getLogger('VisionSystem')

        # Override print function to also log to file in quiet mode
        self.original_print = print

        def logging_print(*args, **kwargs):
            # Call original print
            self.original_print(*args, **kwargs)
            # Also log to file if it's a simple message
            if args:
                message = ' '.join(str(arg) for arg in args)
                # Only log certain types of messages to avoid spam
                if any(keyword in message for keyword in
                       ['[', 'CONFIRMED', 'Still matched', 'REQUEST', 'PERFECT', 'Paused', 'Resumed']):
                    self.logger.info(message)

        # Replace print in quiet mode usage
        import builtins
        builtins.print = logging_print

    def init_modules(self) -> bool:
        """Initialize all modules"""
        if not hasattr(self, 'logger'):
            self.setup_logging()

        self.logger.info("Initializing system modules...")

        try:
            # Initialize camera module
            self.logger.info("Initializing Camera Module...")
            self.camera = CameraModule(self.config_path)
            if not self.camera.init_cameras():
                self.logger.error("Camera Module initialization failed")
                return False
            self.logger.info("Camera Module initialized successfully")

            # Initialize YOLO module
            self.logger.info("Initializing YOLO Recognition Module...")
            self.yolo = YOLOModule(self.config_path)
            if not self.yolo.shelf_model or not self.yolo.area_model:
                self.logger.error("YOLO model loading failed")
                return False
            self.logger.info("YOLO Recognition Module initialized successfully")

            # Initialize UART module
            self.logger.info("Initializing UART Communication Module...")
            self.uart = UARTModule(self.config_path)

            # Ensure serial connection is successful
            if not self.uart.connect():
                self.logger.error("UART serial port connection failed")
                return False

            # Setup signal callbacks
            if self.enable_signal_control:
                self.uart.set_signal_callback(self.on_signal_received)
                # Register callback for requesting last result
                self.uart.set_request_last_result_callback(self.on_request_last_result)
                # Register callback for requesting perfect result
                self.uart.set_request_perfect_result_callback(self.on_request_perfect_result)

                # Ensure initial state is correct
                if self.wait_for_start_signal:
                    self.uart.set_system_state(SystemState.STOPPED)
                    self.logger.info("System initial state set to STOPPED, waiting for start signal")
                else:
                    self.uart.set_system_state(SystemState.RUNNING)
                    self.logger.info("System initial state set to RUNNING, no need to wait for signal")

            self.logger.info("UART Communication Module initialized successfully")

            self.logger.info("All modules initialized")
            return True

        except Exception as e:
            self.logger.error(f"Module initialization failed: {e}")
            return False

    def is_perfect_result(self, result_string: str) -> bool:
        """
        Check if a formatted result string is PERFECT according to rules:
        - Shelf (1..6): must contain all digits 1-6 exactly once, no 'x'
        - Area (a..f): must contain exactly one empty (either '0' or 'x'),
          and the other five are digits 1-6 with no duplicates.
        Any other combination is considered imperfect.
        
        Args:
            result_string: The recognition result string to check
            
        Returns:
            bool: True if the result is perfect, False otherwise
        """
        try:
            parts = result_string.split(",")
            mapping = {}
            for p in parts:
                if ':' not in p:
                    return False
                k, v = p.split(":", 1)
                mapping[k] = v

            # Shelf check: positions 1-6 must contain all digits 1-6 exactly once, no 'x'
            shelf_vals = [mapping.get(str(i), 'x') for i in range(1, 7)]
            if any(v == 'x' for v in shelf_vals):
                return False
            
            try:
                shelf_nums = [int(v) for v in shelf_vals]
            except ValueError:
                return False
                
            if sorted(shelf_nums) != [1, 2, 3, 4, 5, 6]:
                return False

            # Area check: positions a-f must contain exactly one empty and five unique digits 1-6
            area_keys = ['a', 'b', 'c', 'd', 'e', 'f']
            area_vals = [mapping.get(k, 'x') for k in area_keys]
            
            # Count empty positions ('0' or 'x')
            empty_count = sum(1 for v in area_vals if v in ('0', 'x'))
            if empty_count != 1:
                return False
            
            # Get numeric values (excluding empty positions)
            nums = []
            for v in area_vals:
                if v not in ('0', 'x'):
                    try:
                        nums.append(int(v))
                    except ValueError:
                        return False
            
            # Check: exactly 5 numbers, all unique, all in range 1-6
            if len(nums) != 5 or len(set(nums)) != 5:
                return False
            if any(n not in [1, 2, 3, 4, 5, 6] for n in nums):
                return False

            return True
        except Exception as e:
            self.logger.debug(f"Error checking perfect result: {e}")
            return False

    def save_perfect_result_if_qualified(self, result_string: str) -> None:
        """
        Save result as perfect if it meets the perfect criteria
        Note: This should only be called after the result has been confirmed and sent
        
        Args:
            result_string: The recognition result string to check and potentially save
        """
        if result_string and self.is_perfect_result(result_string):
            # Note: This method is called within a result_lock context
            self.last_perfect_result = result_string
            self.stats['perfect_results_found'] += 1
            self.logger.info(f"Perfect result confirmed and saved: {result_string} (Total perfect results: {self.stats['perfect_results_found']})")

    def on_signal_received(self, signal_str: str, old_state: SystemState, new_state: SystemState):
        """
        Signal reception callback function

        Args:
            signal_str: Received signal string
            old_state: Old state
            new_state: New state
        """
        self.stats['signals_processed'] += 1
        self.logger.info(f"Processing signal: '{signal_str}' ({old_state.value} -> {new_state.value})")

        # Execute corresponding operations based on signal type
        if signal_str == self.uart.start_signal:
            self.logger.info("Received start signal, preparing to start detection")
        elif signal_str == self.uart.pause_signal:
            self.logger.info("Received pause signal, will pause after the current cycle is complete")
        elif signal_str == self.uart.resume_signal:
            self.logger.info("Received resume signal, preparing to resume detection")
        elif signal_str == self.uart.stop_signal:
            self.logger.info("Received stop signal, preparing to stop the system")
            self.running = False

    def wait_for_start_signal_with_timeout(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for start signal with timeout support

        Args:
            timeout: Timeout duration (seconds)

        Returns:
            Whether start signal was received
        """
        if not self.enable_signal_control or not self.wait_for_start_signal:
            return True  # Return immediately when signal control is not enabled

        self.uart.set_system_state(SystemState.WAITING_START)
        self.logger.info(f"Waiting for start signal '{self.uart.start_signal}'...")
        print(f"Waiting for start signal '{self.uart.start_signal}'...")

        start_time = time.time()

        while self.uart.get_system_state() == SystemState.WAITING_START and self.running:
            if timeout and (time.time() - start_time) > timeout:
                self.logger.info(f"Waiting for start signal timed out ({timeout} seconds)")
                print(f"Waiting for start signal timed out ({timeout} seconds)")
                return False

            time.sleep(0.1)

        return self.uart.get_system_state() == SystemState.RUNNING

    def on_request_last_result(self):
        """Callback triggered when UART module receives request for last result signal"""
        with self.result_lock:
            if self.last_successful_result:
                self.logger.info(
                    f"Received REQUEST signal, immediately sending last result: {self.last_successful_result}")

                # Show in console for both quiet and normal modes
                print(f"REQUEST -> Sending: {self.last_successful_result}")

                # Send directly in current thread for fastest response
                send_count = self.config['system'].get('result_send_count', 2)
                send_interval = self.config['system'].get('result_send_interval', 0.1)

                successful_sends = 0
                for i in range(send_count):
                    if self.uart.send_message(self.last_successful_result):
                        successful_sends += 1
                    if i < send_count - 1:
                        time.sleep(send_interval)

                self.logger.info(f"REQUEST response complete: {successful_sends}/{send_count} successful")
            else:
                self.logger.warning("Received REQUEST signal, but no last successful result is available")
                print("REQUEST -> No last result available")

    def on_request_perfect_result(self):
        """Callback triggered when UART receives PERFECT result request signal"""
        with self.result_lock:
            if self.last_perfect_result:
                self.logger.info(
                    f"Received PERFECT signal, immediately sending perfect result: {self.last_perfect_result}")
                
                # Output for quiet mode - use Chinese characters as requested
                print("完美数据")  # "Perfect Data" in Chinese for quiet mode
                print(f"PERFECT -> Sending: {self.last_perfect_result}")

                send_count = self.config['system'].get('result_send_count', 2)
                send_interval = self.config['system'].get('result_send_interval', 0.1)

                successful_sends = 0
                for i in range(send_count):
                    if self.uart.send_message(self.last_perfect_result):
                        successful_sends += 1
                    if i < send_count - 1:
                        time.sleep(send_interval)

                self.logger.info(f"PERFECT response complete: {successful_sends}/{send_count} successful")
            else:
                self.logger.warning("Received PERFECT signal, but no perfect result is available yet")
                print("PERFECT -> No perfect result available")

    def single_recognition_with_signal_control(self) -> Dict:
        """Execute single recognition flow (with signal control)"""
        start_time = time.time()
        self.stats['total_runs'] += 1

        try:
            self.logger.info("Starting single recognition process...")

            # Check if detection can be started
            current_state = self.uart.get_system_state()
            can_start = self.uart.can_start_detection()

            if not can_start:
                self.logger.info(f"System state is {current_state.value}, waiting for start signal")

                # If need to wait for start signal, wait here
                if self.enable_signal_control and self.wait_for_start_signal:
                    auto_start_timeout = self.config['system'].get('auto_start_timeout', 30)
                    self.logger.info(f"Waiting for start signal, timeout: {auto_start_timeout} seconds")

                    if not self.wait_for_start_signal_with_timeout(auto_start_timeout):
                        self.logger.info("Timeout, starting detection automatically")
                        self.uart.set_system_state(SystemState.RUNNING)
                    else:
                        self.logger.info("Received start signal, starting detection")
                else:
                    return {'success': False, 'error': 'Waiting for start signal', 'waiting': True}

            # Confirm state again
            final_state = self.uart.get_system_state()
            if final_state != SystemState.RUNNING:
                self.uart.set_system_state(SystemState.RUNNING)

            # 1. Get image data
            self.logger.info("Acquiring image data...")
            # Read capture count and interval from configuration file
            capture_count = self.config['camera'].get('capture_count', 1)
            interval_ms = self.config['camera'].get('default_capture_interval_ms', 200)

            camera_data = self.camera.get_images_for_yolo(
                count=capture_count,
                interval_ms=interval_ms
            )

            if not camera_data['success']:
                error_msg = f"Image acquisition failed: {camera_data.get('error')}"
                self.logger.error(error_msg)
                self.stats['failed_runs'] += 1
                return {'success': False, 'error': error_msg}

            # Check if pause is needed
            if self.uart.should_pause_detection():
                self.logger.info("Pause signal detected, pausing after this detection cycle")
                self.stats['paused_runs'] += 1
                return {'success': False, 'error': 'Detection paused', 'paused': True}

            # Update current images (for visualization display)
            if camera_data['shelf_images']:
                self.current_shelf_image = camera_data['shelf_images'][0].copy()
            if camera_data['area_images']:
                self.current_area_image = camera_data['area_images'][0].copy()

            # Optional: save debug images
            if self.config['system'].get('save_debug_images', False):
                self.camera.save_images(
                    camera_data['shelf_images'],
                    camera_data['area_images'],
                    f"run_{self.stats['total_runs']}"
                )

            # 2. YOLO recognition
            self.logger.info("Performing YOLO recognition...")
            recognition_result = self.yolo.process_camera_data(camera_data)

            if not recognition_result['success']:
                error_msg = f"YOLO recognition failed: {recognition_result.get('error')}"
                self.logger.error(error_msg)
                self.stats['failed_runs'] += 1
                return {'success': False, 'error': error_msg}

            result_string = recognition_result['result_string']
            self.logger.info(f"YOLO recognition complete: {result_string}")

            # Check continuous matching requirements
            consensus_method = recognition_result.get('consensus_method', 'voting')
            if consensus_method == "consistency":
                if not recognition_result.get('ready_to_send', False):
                    # Not reached continuous matching requirement, record but don't send
                    consecutive_info = f"{recognition_result.get('consecutive_count', 0)}/{recognition_result.get('consecutive_required', 2)}"
                    self.logger.info(f"Consecutive match progress: {consecutive_info}, waiting for more matches...")

                    # Update detection results for display, but mark as waiting state
                    self.update_detection_results(recognition_result)
                    self.current_result_string = f"{result_string} (Waiting: {consecutive_info})"

                    return {
                        'success': True,
                        'waiting_for_consistency': True,
                        'result_string': result_string,
                        'consecutive_count': recognition_result.get('consecutive_count', 0),
                        'consecutive_required': recognition_result.get('consecutive_required', 2)
                    }

            # Update detection results (for visualization display)
            self.update_detection_results(recognition_result)
            self.current_result_string = result_string

            # 3. Serial sending (repeat multiple times)
            send_success = False
            if result_string:
                send_count = self.config['system'].get('result_send_count', 5)
                send_interval = self.config['system'].get('result_send_interval', 0.5)

                self.logger.info(f"Sending recognition result ({send_count} repetitions)...")

                success_count = 0
                for i in range(send_count):
                    if self.uart.send_result(result_string):
                        success_count += 1

                    if i < send_count - 1:
                        time.sleep(send_interval)

                send_success = success_count > 0
                self.logger.info(f"Sending complete: {success_count}/{send_count} successful")

            # 4. Update statistics
            self.stats['successful_runs'] += 1
            self.stats['last_result'] = result_string
            self.stats['last_run_time'] = datetime.now()

            elapsed_time = time.time() - start_time
            self.logger.info(f"Single recognition complete, time elapsed: {elapsed_time:.2f} seconds")

            # Save this successful result for STM32 random requests
            with self.result_lock:
                self.last_successful_result = result_string
                # Also check and save as perfect result if it meets criteria
                self.save_perfect_result_if_qualified(result_string)
                self.logger.info(f"Saved latest successful result: {self.last_successful_result}")

            return {
                'success': True,
                'result_string': result_string,
                'recognition_result': recognition_result,
                'send_success': send_success,
                'elapsed_time': elapsed_time
            }

        except Exception as e:
            error_msg = f"Exception in recognition process: {e}"
            self.logger.error(error_msg)
            self.stats['failed_runs'] += 1
            return {'success': False, 'error': error_msg}

    def update_detection_results(self, recognition_result: Dict):
        """Update detection results for visualization display"""
        try:
            # Clear previous detection results
            self.current_detections = {'shelf': [], 'area': []}

            # Update shelf detection results
            if 'shelf_results' in recognition_result and recognition_result['shelf_results'].get('success'):
                shelf_detections = recognition_result['shelf_results'].get('detections', [])
                for detection in shelf_detections:
                    if 'bbox' in detection and 'confidence' in detection:
                        self.current_detections['shelf'].append({
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence'],
                            'number': detection.get('number', '?'),
                            'position': detection.get('position', '?'),
                            'inference_size': detection.get('inference_size', 'N/A')
                        })

            # Update area detection results
            if 'area_results' in recognition_result and recognition_result['area_results'].get('success'):
                area_detections = recognition_result['area_results'].get('detections', [])
                for detection in area_detections:
                    if 'bbox' in detection and 'confidence' in detection:
                        self.current_detections['area'].append({
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence'],
                            'number': detection.get('number', '?'),
                            'position': detection.get('position', '?'),
                            'inference_size': detection.get('inference_size', 'N/A')
                        })

        except Exception as e:
            self.logger.error(f"Failed to update detection results: {e}")

    def draw_detections_on_image(self, image: np.ndarray, detections: List[Dict], image_type: str) -> np.ndarray:
        """Draw detection results on image (fixed version)"""
        if image is None:
            return None

        result_image = image.copy()
        visual_config = self.config.get('visual', {})

        font_scale = visual_config.get('font_scale', 0.8)
        line_thickness = visual_config.get('line_thickness', 2)
        text_color = tuple(visual_config.get('text_color', [0, 255, 0]))
        bbox_color = tuple(visual_config.get('bbox_color', [255, 0, 0]))
        conf_threshold = visual_config.get('confidence_threshold_display', 0.3)

        # Get original image dimensions (for bbox scaling)
        original_height = 1080  # From config
        original_width = 1920  # From config
        display_height, display_width = result_image.shape[:2]

        # Calculate scaling factors
        scale_x = display_width / original_width
        scale_y = display_height / original_height

        for detection in detections:
            confidence = detection.get('confidence', 0)
            if confidence < conf_threshold:
                continue

            bbox = detection.get('bbox', [])
            if len(bbox) != 4:
                continue

            # Scale bbox coordinates to display image size
            x1, y1, x2, y2 = bbox
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, display_width - 1))
            y1 = max(0, min(y1, display_height - 1))
            x2 = max(0, min(x2, display_width - 1))
            y2 = max(0, min(y2, display_height - 1))

            number = detection.get('number', '?')
            position = detection.get('position', '?')
            inference_size = detection.get('inference_size', 'N/A')

            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), bbox_color, line_thickness)

            # Draw label (including inference resolution information)
            label = f"{position}:{number} ({confidence:.2f}) @{inference_size}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)[0]

            # Draw label background
            cv2.rectangle(result_image,
                          (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1),
                          bbox_color, -1)

            # Draw label text
            cv2.putText(result_image, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, text_color, line_thickness)

        # Add type identifier and system state to image
        type_label = f"{image_type.upper()} Camera - State: {self.uart.get_system_state().value}"
        cv2.putText(result_image, type_label,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, text_color, 2)

        return result_image

    def continuous_matching_mode(self, quiet_mode=False):
        """Continuous matching mode - completely silent version with perfect result tracking"""
        if not quiet_mode:
            print("Continuous Matching Mode")
        else:
            print("Continuous Matching - Quiet Mode")

        if quiet_mode:
            # Create completely silent YOLO output wrapper function
            def silent_yolo_process(camera_data):
                import io, sys, os
                # Save original output
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                # Redirect to null device
                try:
                    devnull = open(os.devnull, 'w')
                    sys.stdout = devnull
                    sys.stderr = devnull

                    # Execute YOLO processing
                    result = self.yolo.process_camera_data(camera_data)
                finally:
                    # Restore output
                    if 'devnull' in locals():
                        devnull.close()
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

                return result

            # Create silent camera capture function
            def silent_camera_capture():
                import io, sys, os
                old_stdout = sys.stdout
                try:
                    devnull = open(os.devnull, 'w')
                    sys.stdout = devnull
                    result = self.camera.get_images_for_yolo(count=1, interval_ms=0)
                finally:
                    if 'devnull' in locals():
                        devnull.close()
                    sys.stdout = old_stdout
                return result

        self.running = True
        self.stats['start_time'] = datetime.now()

        # Configuration
        cfg = self.config.get('yolo', {}).get('continuous_matching', {})
        required_matches = cfg.get('required_matches', 2)
        check_interval = cfg.get('check_interval', 0.2)

        tracker = ContinuousMatchingTracker(required_matches)

        if quiet_mode:
            print(f"Config: {required_matches} matches, {check_interval}s interval")
        else:
            self.logger.info(f"Config: {required_matches} matches, {check_interval}s interval")

        # Signal control
        if self.enable_signal_control and not self.wait_for_start_signal:
            if quiet_mode:
                import os
                old_stdout = sys.stdout
                try:
                    devnull = open(os.devnull, 'w')
                    sys.stdout = devnull
                    self.uart.set_system_state(SystemState.RUNNING)
                finally:
                    if 'devnull' in locals():
                        devnull.close()
                    sys.stdout = old_stdout
            else:
                self.uart.set_system_state(SystemState.RUNNING)

        # Verify REQUEST and PERFECT signal callbacks are set
        if not hasattr(self.uart, 'request_last_result_callback') or self.uart.request_last_result_callback is None:
            if quiet_mode:
                print("Warning: REQUEST signal callback not set, setting it now...")
            else:
                self.logger.warning("REQUEST signal callback not set, setting it now...")
            self.uart.set_request_last_result_callback(self.on_request_last_result)
        
        if hasattr(self.uart, 'set_request_perfect_result_callback'):
            self.uart.set_request_perfect_result_callback(self.on_request_perfect_result)

        run_num = 0

        try:
            while self.running:
                run_num += 1
                t_start = time.time()

                # Check for pause signal before starting recognition
                if self.uart.should_pause_detection():
                    if quiet_mode:
                        print("Paused - waiting for resume signal...")
                    else:
                        self.logger.info("Detection paused, waiting for resume signal...")

                    # Wait for resume signal
                    while self.uart.should_pause_detection() and self.running:
                        time.sleep(0.5)

                    if not self.running:
                        break

                    if quiet_mode:
                        print("Resumed - continuing recognition...")
                    else:
                        self.logger.info("Detection resumed, continuing...")

                # Get images
                if quiet_mode:
                    camera_data = silent_camera_capture()
                else:
                    self.logger.info(f"Run #{run_num} [{tracker.get_status()['current_count']}/{required_matches}]")
                    camera_data = self.camera.get_images_for_yolo(count=1, interval_ms=0)

                if not camera_data['success']:
                    tracker.reset()
                    time.sleep(check_interval)
                    continue

                # YOLO recognition
                orig_min = self.yolo.config["yolo"]["min_consensus_count"]
                self.yolo.config["yolo"]["min_consensus_count"] = 1

                if quiet_mode:
                    recognition_result = silent_yolo_process(camera_data)
                else:
                    recognition_result = self.yolo.process_camera_data(camera_data)

                self.yolo.config["yolo"]["min_consensus_count"] = orig_min

                if not recognition_result['success']:
                    tracker.reset()
                    time.sleep(check_interval)
                    continue

                result_string = recognition_result['result_string']
                t_elapsed = time.time() - t_start

                # Output results
                if quiet_mode:
                    print(f"[{run_num:3d}] {result_string} | {t_elapsed:.2f}s")
                else:
                    self.logger.info(f"Run #{run_num}: {result_string} ({t_elapsed:.2f}s)")

                # Check matching
                match_status = tracker.add_result(result_string)

                if not quiet_mode:
                    self.logger.info(f"Progress: {match_status['current_count']}/{match_status['required_count']}")

                # Send only when matched (confirmed)
                if match_status['matched']:
                    send_count = self.config['system'].get('result_send_count', 2)

                    if match_status['is_new_confirmation']:
                        if quiet_mode:
                            print(f"     CONFIRMED after {required_matches} matches! -> Sent {send_count} times")
                        else:
                            self.logger.info(f"CONFIRMED: {result_string}")
                    else:
                        if quiet_mode:
                            print(
                                f"     Still matched ({match_status['current_count']} times) -> Sent {send_count} times")
                        else:
                            self.logger.info(f"Still matched ({match_status['current_count']} times)")

                    # Silent sending
                    if quiet_mode:
                        import os
                        old_stdout = sys.stdout
                        try:
                            devnull = open(os.devnull, 'w')
                            sys.stdout = devnull

                            for i in range(send_count):
                                self.uart.send_result(result_string)
                                if i < send_count - 1:
                                    time.sleep(0.1)
                        finally:
                            if 'devnull' in locals():
                                devnull.close()
                            sys.stdout = old_stdout
                    else:
                        for i in range(send_count):
                            self.uart.send_result(result_string)
                            if i < send_count - 1:
                                time.sleep(0.1)
                        self.logger.info(f"Sent {send_count} times")

                    self.stats['successful_runs'] += 1
                    # Save confirmed result and check for perfect result
                    with self.result_lock:
                        self.last_successful_result = result_string
                        # Only save as perfect result after confirmation and sending
                        self.save_perfect_result_if_qualified(result_string)

                time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            self.running = False
            if quiet_mode:
                perfect_count = self.stats['perfect_results_found']
                print(f"Total: {run_num}, Confirmed: {self.stats['successful_runs']}, Perfect: {perfect_count}")

    def continuous_mode_with_signal_control(self, quiet_mode=False):
        """Continuous recognition mode with signal control and perfect result tracking"""
        if not quiet_mode:
            self.logger.info("Starting continuous recognition mode with signal control...")
        else:
            print("Continuous Mode (voting) - Starting...")

        self.running = True
        self.stats['start_time'] = datetime.now()

        run_interval = self.config['system'].get('run_interval', 0.3)
        check_interval = 0.2  # Define check_interval for quiet mode

        # Create silent functions for quiet mode
        if quiet_mode:
            def silent_yolo_process(camera_data):
                import io, sys, os
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                try:
                    devnull = open(os.devnull, 'w')
                    sys.stdout = devnull
                    sys.stderr = devnull
                    result = self.yolo.process_camera_data(camera_data)
                finally:
                    if 'devnull' in locals():
                        devnull.close()
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr
                return result

            def silent_camera_capture():
                import io, sys, os
                old_stdout = sys.stdout
                try:
                    devnull = open(os.devnull, 'w')
                    sys.stdout = devnull
                    result = self.camera.get_images_for_yolo()
                finally:
                    if 'devnull' in locals():
                        devnull.close()
                    sys.stdout = old_stdout
                return result

        try:
            # If signal control is enabled and need to wait for start signal
            if self.enable_signal_control and self.wait_for_start_signal:
                auto_start_timeout = self.config['system'].get('auto_start_timeout', 30)
                if not quiet_mode:
                    self.logger.info(f"Waiting for start signal or timeout ({auto_start_timeout} seconds)...")

                if not self.wait_for_start_signal_with_timeout(auto_start_timeout):
                    if self.uart.get_system_state() == SystemState.WAITING_START:
                        if not quiet_mode:
                            self.logger.info("Waiting for start signal timed out, starting detection automatically")
                        self.uart.set_system_state(SystemState.RUNNING)
            elif self.enable_signal_control:
                # Signal control enabled but not waiting for start signal, set directly to running state
                if quiet_mode:
                    import os
                    old_stdout = sys.stdout
                    try:
                        devnull = open(os.devnull, 'w')
                        sys.stdout = devnull
                        self.uart.set_system_state(SystemState.RUNNING)
                    finally:
                        if 'devnull' in locals():
                            devnull.close()
                        sys.stdout = old_stdout
                else:
                    self.uart.set_system_state(SystemState.RUNNING)

            while self.running:
                if not quiet_mode:
                    self.logger.info(f"Starting recognition run #{self.stats['total_runs'] + 1}")

                # Check for pause signal before starting recognition
                if self.uart.should_pause_detection():
                    if quiet_mode:
                        print("Paused - waiting for resume signal...")
                    else:
                        self.logger.info("Detection paused, waiting for resume signal...")

                    # Wait for resume signal
                    while self.uart.should_pause_detection() and self.running:
                        time.sleep(0.5)

                    if not self.running:
                        break

                    if quiet_mode:
                        print("Resumed - continuing recognition...")
                    else:
                        self.logger.info("Detection resumed, continuing...")
                    continue

                # Execute single recognition with silent processing in quiet mode
                if quiet_mode:
                    # Silent mode: capture and process without output
                    t_start = time.time()
                    camera_data = silent_camera_capture()

                    if not camera_data['success']:
                        time.sleep(check_interval)
                        continue

                    recognition_result = silent_yolo_process(camera_data)

                    if not recognition_result['success']:
                        time.sleep(check_interval)
                        continue

                    result_string = recognition_result['result_string']
                    elapsed_time = time.time() - t_start

                    # Silent sending
                    old_stdout = sys.stdout
                    try:
                        devnull = open(os.devnull, 'w')
                        sys.stdout = devnull
                        send_count = self.config['system'].get('result_send_count', 2)
                        for i in range(send_count):
                            self.uart.send_result(result_string)
                            if i < send_count - 1:
                                time.sleep(0.1)
                    finally:
                        if 'devnull' in locals():
                            devnull.close()
                        sys.stdout = old_stdout

                    # Update stats and save results only after successful sending
                    self.stats['total_runs'] += 1
                    self.stats['successful_runs'] += 1
                    with self.result_lock:
                        self.last_successful_result = result_string
                        # Only save as perfect result after successful sending
                        self.save_perfect_result_if_qualified(result_string)

                    # Simple output like continuous-matching mode
                    print(f"[{self.stats['total_runs']:3d}] {result_string} | {elapsed_time:.2f}s -> Sent")

                else:
                    # Normal mode: full output
                    result = self.single_recognition_with_signal_control()

                    if result['success']:
                        if result.get('waiting_for_consistency'):
                            # Waiting for consecutive matches, shorten wait time
                            consecutive_info = f"{result.get('consecutive_count', 0)}/{result.get('consecutive_required', 2)}"
                            self.logger.info(f"Waiting for consecutive match: {consecutive_info}")
                            time.sleep(0.5)  # Shorten wait time for quick next recognition
                            continue
                        else:
                            # Normal recognition success, can send
                            self.logger.info(f"Recognition successful: {result['result_string']}")
                    elif result.get('waiting'):
                        self.logger.info("Waiting for start signal...")
                        time.sleep(1)  # Shorter interval for waiting signal
                        continue
                    elif result.get('paused'):
                        self.logger.info("Detection paused, waiting for resume signal...")
                        # Shorter check interval during pause
                        while self.uart.should_pause_detection() and self.running:
                            time.sleep(0.5)
                        continue
                    else:
                        self.logger.error(f"Recognition failed: {result.get('error')}")

                    # Display statistics only every 10 runs to reduce output
                    if self.stats['total_runs'] % 10 == 0:
                        self.print_stats()

                # Wait for next run
                if self.running and not self.uart.should_pause_detection():
                    run_interval = self.config['system'].get('run_interval', 10)
                    if not quiet_mode:
                        self.logger.info(f"Waiting {run_interval} seconds to continue...")

                # Use new while loop to support fractional second waits and maintain interruptibility
                wait_start_time = time.time()
                while (time.time() - wait_start_time) < run_interval:
                    # Check for pause or stop signals
                    if not self.running or self.uart.should_pause_detection():
                        break
                    # Sleep in small increments to ensure timely signal response
                    time.sleep(0.1)

        except KeyboardInterrupt:
            if not quiet_mode:
                self.logger.info("User interrupted continuous mode")
            else:
                print("\nInterrupted")
        except Exception as e:
            if quiet_mode:
                print(f"Error: {e}")
            else:
                self.logger.error(f"Exception in continuous mode: {e}")
        finally:
            self.running = False
            if quiet_mode:
                perfect_count = self.stats['perfect_results_found']
                print(f"Total: {self.stats['total_runs']}, Successful: {self.stats['successful_runs']}, Perfect: {perfect_count}")

    def visual_mode_with_signal_control(self):
        """Visual mode with signal control (integrated all features: background recognition, frame rate control, manual capture)"""
        self.logger.info("Starting Visual Mode with signal control...")
        self.running = True
        self.display_active = True
        self.stats['start_time'] = datetime.now()

        # Calculate delay per frame based on configuration to control frame rate
        if self.preview_fps > 0:
            delay_ms = int(1000 / self.preview_fps)
        else:
            delay_ms = 1
        print(f"Preview FPS limit set to: {self.preview_fps}, frame delay: {delay_ms}ms")

        print("\nVisual Mode")
        print("Press ESC to quit, SPACE to manually trigger recognition, 'c' to capture image")

        # Background recognition thread definition
        def recognition_worker():
            run_interval = self.config['system'].get('run_interval', 10)
            while self.running:
                try:
                    result = self.single_recognition_with_signal_control()
                    if result['success']:
                        self.logger.info(f"Recognition successful: {result['result_string']}")
                    elif result.get('waiting') or result.get('paused'):
                        time.sleep(1)
                        continue
                    else:
                        self.logger.error(f"Recognition failed: {result.get('error')}")

                    # Wait interval
                    wait_start_time = time.time()
                    while (time.time() - wait_start_time) < run_interval:
                        if not self.running:
                            break
                        time.sleep(0.1)

                except Exception as e:
                    self.logger.error(f"Exception in recognition thread: {e}")
                    time.sleep(1)

        # Start background recognition thread
        self.recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
        self.recognition_thread.start()

        try:
            # Main loop for display and key handling
            while self.running:
                # Get real-time frames from cameras for display
                if self.camera and self.camera.shelf_camera and self.camera.area_camera:
                    ret1, frame1 = self.camera.shelf_camera.read()
                    ret2, frame2 = self.camera.area_camera.read()

                    if ret1 and ret2:
                        display_width = self.config['visual'].get('display_width', 480)
                        display_height = self.config['visual'].get('display_height', 270)

                        frame1_display = cv2.resize(frame1, (display_width, display_height))
                        frame2_display = cv2.resize(frame2, (display_width, display_height))

                        # Draw latest detection boxes recognized by background thread on frames
                        if self.current_shelf_image is not None:
                            detection_image1 = self.draw_detections_on_image(
                                frame1_display, self.current_detections['shelf'], 'Shelf')
                            if detection_image1 is not None:
                                frame1_display = detection_image1

                        if self.current_area_image is not None:
                            detection_image2 = self.draw_detections_on_image(
                                frame2_display, self.current_detections['area'], 'Area')
                            if detection_image2 is not None:
                                frame2_display = detection_image2

                        # Combine two frames for display
                        combined_preview = np.hstack((frame1_display, frame2_display))
                        cv2.imshow("Vision System Preview", combined_preview)

                # Handle keys, using calculated delay to control frame rate
                key = cv2.waitKey(delay_ms) & 0xFF

                if key == 27:  # ESC key to exit
                    print("ESC pressed, exiting...")
                    break
                elif key == ord(' '):  # Space key to manually trigger recognition
                    print("Manually triggering recognition...")
                    threading.Thread(target=self.single_recognition_with_signal_control, daemon=True).start()

                elif key == ord('c'):  # c key to capture
                    capture_dir = "visual_mode_captured"
                    os.makedirs(capture_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")

                    # Read latest frames directly from camera objects to ensure highest resolution
                    ret_shelf, frame_shelf_cap = self.camera.shelf_camera.read()
                    ret_area, frame_area_cap = self.camera.area_camera.read()

                    if ret_shelf:
                        shelf_filename = os.path.join(capture_dir, f"shelf_{timestamp}.jpg")
                        cv2.imwrite(shelf_filename, frame_shelf_cap)
                        print(f"Photo saved: {shelf_filename}")

                    if ret_area:
                        area_filename = os.path.join(capture_dir, f"area_{timestamp}.jpg")
                        cv2.imwrite(area_filename, frame_area_cap)
                        print(f"Photo saved: {area_filename}")

        except Exception as e:
            self.logger.error(f"Visual mode exception: {e}")
        finally:
            self.running = False
            self.display_active = False
            cv2.destroyAllWindows()

    def debug_mode(self):
        """Debug mode - display detailed recognition process and results"""
        self.logger.info("Starting debug mode...")
        self.running = True
        self.stats['start_time'] = datetime.now()

        print("\nDebug Mode")
        print("=" * 60)
        print("Features:")
        print("- Detailed display of each recognition step")
        print("- Save images from each recognition")
        print("- Display detection results for each image")
        print("- Display inference resolution and position mapping information")
        print("- Display voting and consensus algorithm process")
        print("- Display final result generation process")
        print("- Track perfect results automatically")
        if self.enable_signal_control:
            print(f"- Supports signal control (Start signal: '{self.uart.start_signal}')")
        print("- Press Enter to continue to the next recognition, 'q' to quit")
        print("=" * 60)

        debug_count = 1

        try:
            while self.running:
                print(f"\n{'=' * 80}")
                print(f"Debug Run #{debug_count}")
                print(f"{'=' * 80}")

                # Wait for user input
                user_input = input("Press Enter to start recognition, enter 'q' to quit: ").strip().lower()
                if user_input == 'q':
                    break

                # Execute detailed recognition flow
                self.detailed_recognition_debug()

                debug_count += 1

        except KeyboardInterrupt:
            self.logger.info("User interrupted debug mode")
        except Exception as e:
            self.logger.error(f"Exception in debug mode: {e}")
        finally:
            self.running = False

    def detailed_recognition_debug(self):
        """Detailed recognition debug flow"""
        start_time = time.time()

        print("\nStep 1: Signal Control Check")
        print("-" * 40)

        # Check signal control state
        if self.enable_signal_control:
            print(f"Signal control enabled")
            print(f"  Current system state: {self.uart.get_system_state().value}")
            print(f"  Start signal: '{self.uart.start_signal}'")
            if self.enable_pause_control:
                print(f"  Pause signal: '{self.uart.pause_signal}'")
                print(f"  Resume signal: '{self.uart.resume_signal}'")
            print(f"  Stop signal: '{self.uart.stop_signal}'")
            print(f"  Request last result signal: '{self.uart.request_last_result_signal}'")
            print(f"  Request perfect result signal: '{self.uart.request_perfect_result_signal}'")

            # Check if need to wait for start signal
            if not self.uart.can_start_detection():
                if self.wait_for_start_signal:
                    print(f"Waiting for start signal...")
                    auto_start_timeout = self.config['system'].get('auto_start_timeout', 30)
                    if not self.wait_for_start_signal_with_timeout(auto_start_timeout):
                        print(f"Timeout, starting detection automatically")
                        self.uart.set_system_state(SystemState.RUNNING)
                else:
                    print(f"System state does not allow detection to start: {self.uart.get_system_state().value}")
                    return
        else:
            print(f"Signal control not enabled")

        print("\nStep 2: Image Acquisition")
        print("-" * 40)

        # Get image data
        capture_count = self.config['camera'].get('capture_count', 1)
        interval_ms = self.config['camera'].get('default_capture_interval_ms', 200)
        camera_data = self.camera.get_images_for_yolo(count=capture_count, interval_ms=interval_ms)

        if not camera_data['success']:
            print(f"Image acquisition failed: {camera_data.get('error')}")
            return

        print(f"Successfully acquired {camera_data['count']} images")
        print(f"  Shelf images: {len(camera_data['shelf_images'])}")
        print(f"  Area images: {len(camera_data['area_images'])}")

        # Save debug images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_prefix = f"debug_{timestamp}"
        self.camera.save_images(
            camera_data['shelf_images'],
            camera_data['area_images'],
            debug_prefix
        )
        print(f"  Debug images saved: {debug_prefix}_*.jpg")

        print("\nStep 3: YOLO Recognition")
        print("-" * 40)

        # Display inference configuration
        print(f"Inference Configuration:")
        print(f"  Shelf inference resolution: {self.yolo.shelf_inference_size}")
        print(f"  Area inference resolution: {self.yolo.area_inference_size}")
        print(f"  Position mapping method: Absolute Position Mapping")

        # Detailed YOLO recognition process
        recognition_result = self.detailed_yolo_debug(camera_data)

        if not recognition_result['success']:
            print(f"YOLO recognition failed: {recognition_result.get('error')}")
            return

        print("\nStep 4: Result Processing and Sending")
        print("-" * 40)

        result_string = recognition_result['result_string']
        print(f"Final result string: {result_string}")

        # Serial sending
        send_count = self.config['system'].get('result_send_count', 5)
        send_interval = self.config['system'].get('result_send_interval', 0.5)

        print(f"Serial sending configuration: {send_count} repetitions, interval {send_interval} seconds")

        success_count = 0
        for i in range(send_count):
            if self.uart.send_result(result_string):
                success_count += 1
                print(f"  Attempt {i + 1} sent successfully")
            else:
                print(f"  Attempt {i + 1} failed to send")

            if i < send_count - 1:
                time.sleep(send_interval)

        print(f"Sending result: {success_count}/{send_count} successful")

        # Only check and save perfect result after successful sending
        if success_count > 0:
            print("\nStep 5: Perfect Result Check (After Confirmation)")
            print("-" * 40)
            
            is_perfect = self.is_perfect_result(result_string)
            print(f"Recognition result: {result_string}")
            print(f"Perfect result check: {'YES - PERFECT!' if is_perfect else 'No - not perfect'}")
            
            if is_perfect:
                print(f"  ✓ Shelf (1-6): Contains all digits 1-6, no duplicates, no 'x'")
                print(f"  ✓ Area (a-f): Contains exactly one empty position and five unique digits 1-6")
                print(f"  ✓ Result confirmed and sent successfully")
                with self.result_lock:
                    self.save_perfect_result_if_qualified(result_string)
            else:
                print(f"  ✗ Does not meet perfect result criteria")

            print(f"Total perfect results found so far: {self.stats['perfect_results_found']}")
        else:
            print("\nStep 5: Perfect Result Check Skipped")
            print("-" * 40)
            print("Perfect result check skipped because sending failed")

        elapsed_time = time.time() - start_time
        print(f"\nTotal time elapsed: {elapsed_time:.2f} seconds")

        # Update statistics only if sending was successful
        if success_count > 0:
            self.stats['total_runs'] += 1
            self.stats['successful_runs'] += 1
            self.stats['last_result'] = result_string
            self.stats['last_run_time'] = datetime.now()

            # Save successful result
            with self.result_lock:
                self.last_successful_result = result_string

    def detailed_yolo_debug(self, camera_data: Dict) -> Dict:
        """Detailed YOLO recognition debug"""
        try:
            shelf_images = camera_data['shelf_images']
            area_images = camera_data['area_images']

            print(f"  Recognizing {len(shelf_images)} shelf images...")

            # Shelf recognition
            print("\n  Shelf Recognition Details:")
            shelf_results = self.yolo.detect_shelf_numbers(shelf_images)

            if shelf_results['success']:
                print(f"    Shelf recognition successful")
                print(f"    Inference resolution: {shelf_results.get('inference_size', 'N/A')}")

                # Display detection results for each image
                for i, detection in enumerate(shelf_results.get('detections', [])):
                    img_idx = detection.get('image_index', '?')
                    position = detection.get('position', '?')
                    number = detection.get('number', '?')
                    confidence = detection.get('confidence', 0)
                    inference_size = detection.get('inference_size', 'N/A')
                    center = detection.get('center', [0, 0])
                    norm_x = center[0] / detection.get('image_width', 1920)
                    norm_y = center[1] / detection.get('image_height', 1080)
                    print(
                        f"      Image {img_idx + 1}: Position {position} -> Number {number} (Confidence: {confidence:.3f}) Coords: ({norm_x:.3f}, {norm_y:.3f}) @{inference_size}")

                # Display consensus results
                consensus = shelf_results.get('consensus', {})
                if consensus:
                    print(f"    Shelf Consensus Results:")
                    for position in sorted(consensus.keys()):
                        info = consensus[position]
                        print(
                            f"      Position {position}: Number {info['number']} (Votes: {info['vote_count']}/{info['total_detections']}, Confidence: {info['confidence']:.3f})")
                else:
                    print(f"    No shelf results passed the consensus check")

            # Area recognition
            print(f"\n  Area Recognition Details:")
            area_results = self.yolo.detect_area_numbers(area_images)

            if area_results['success']:
                print(f"    Area recognition successful")
                print(f"    Inference resolution: {area_results.get('inference_size', 'N/A')}")

                # Display detection results for each image
                for i, detection in enumerate(area_results.get('detections', [])):
                    img_idx = detection.get('image_index', '?')
                    position = detection.get('position', '?')
                    number = detection.get('number', '?')
                    confidence = detection.get('confidence', 0)
                    inference_size = detection.get('inference_size', 'N/A')
                    center = detection.get('center', [0, 0])
                    norm_x = center[0] / detection.get('image_width', 1920)
                    norm_y = center[1] / detection.get('image_height', 1080)
                    number_str = "Empty" if number == 0 else f"Number {number}"
                    print(
                        f"      Image {img_idx + 1}: Position {position} -> {number_str} (Confidence: {confidence:.3f}) Coords: ({norm_x:.3f}, {norm_y:.3f}) @{inference_size}")

                # Display consensus results
                consensus = area_results.get('consensus', {})
                if consensus:
                    print(f"    Area Consensus Results:")
                    for position in sorted(consensus.keys()):
                        info = consensus[position]
                        number_str = "Empty" if info['number'] == 0 else f"Number {info['number']}"
                        print(
                            f"      Position {position}: {number_str} (Votes: {info['vote_count']}/{info['total_detections']}, Confidence: {info['confidence']:.3f})")
                else:
                    print(f"    No area results passed the consensus check")

            # Generate final result
            print(f"\n  Result String Generation:")
            result_string = self.yolo.format_result_string(shelf_results, area_results)
            print(f"    Full result: {result_string}")

            # Parse result display
            result_parts = result_string.split(',')
            shelf_parts = [p for p in result_parts if p.split(':')[0].isdigit()]
            area_parts = [p for p in result_parts if p.split(':')[0].isalpha()]

            print(f"    Shelf part: {','.join(shelf_parts)}")
            print(f"    Area part: {','.join(area_parts)}")

            return {
                'success': True,
                'shelf_results': shelf_results,
                'area_results': area_results,
                'result_string': result_string
            }

        except Exception as e:
            print(f"    YOLO recognition exception: {e}")
            return {'success': False, 'error': str(e)}

    def test_signal_control(self):
        """Test signal control functionality"""
        print("\n" + "=" * 60)
        print("Signal Control Function Test")
        print("=" * 60)

        if not self.enable_signal_control:
            print("Signal control not enabled, please check the configuration file")
            return

        print(f"Signal control enabled")
        print(f"Start signal: '{self.uart.start_signal}'")
        print(f"Wait for start signal: {self.wait_for_start_signal}")
        print(f"Request last result signal: '{self.uart.request_last_result_signal}'")
        print(f"Request perfect result signal: '{self.uart.request_perfect_result_signal}'")
        print(f"Current system state: {self.uart.get_system_state().value}")

        # Test serial connection status
        if self.uart.is_connected:
            print(f"Serial port connected: {self.uart.config['uart']['port']}")
        else:
            print(f"Serial port not connected")
            if self.uart.connect():
                print(f"Serial port reconnected successfully")
            else:
                print(f"Failed to reconnect to serial port")
                return

        # Test signal listening
        print(f"\nTesting Signal Listener:")
        print(f"   Signal listener thread running: {self.uart.signal_running}")

        if not self.uart.signal_running:
            print(f"   Signal listener thread not running, attempting to start...")
            self.uart.start_signal_listener()
            print(f"   Signal listener thread status: {self.uart.signal_running}")

        # Display signal statistics
        stats = self.uart.get_stats()
        print(f"\nSignal Statistics:")
        print(f"   Signals received: {stats['signals_received']}")
        print(f"   Last signal: {stats['last_signal']}")
        print(f"   Last signal time: {stats['last_signal_time']}")

        print(f"\nTest Suggestions:")
        print(f"   1. Ensure only one program is using the serial port")
        print(f"   2. Do not run uart_module.py --listen simultaneously")
        print(f"   3. Send signal format: echo '{self.uart.start_signal}' > {self.uart.config['uart']['port']}")
        print(f"   4. Test REQUEST: echo '{self.uart.request_last_result_signal}' > {self.uart.config['uart']['port']}")
        print(f"   5. Test PERFECT: echo '{self.uart.request_perfect_result_signal}' > {self.uart.config['uart']['port']}")
        print(f"   6. Check serial port permissions: sudo chmod 666 {self.uart.config['uart']['port']}")

    def test_all_modules(self):
        """Test all modules"""
        print("\n" + "=" * 60)
        print("Intelligent Delivery Robot Vision Recognition System - Full Test")
        print("=" * 60)

        # 1. Camera module test
        print("\n1. Camera Module Test:")
        try:
            cameras = self.camera.detect_cameras()
            if cameras:
                print(f"   Detected {len(cameras)} cameras: {cameras}")
                print(f"   Mode: {'Single camera' if self.camera.single_camera_mode else 'Dual camera'}")
            else:
                print(f"   No cameras detected")
        except Exception as e:
            print(f"   Camera test failed: {e}")

        # 2. YOLO module test
        print("\n2. YOLO Recognition Module Test:")
        try:
            print(f"   Device: {self.yolo.device}")
            print(f"   Shelf model: {'Loaded' if self.yolo.shelf_model else 'Not loaded'}")
            print(f"   Area model: {'Loaded' if self.yolo.area_model else 'Not loaded'}")
            print(f"   Shelf inference resolution: {self.yolo.shelf_inference_size}")
            print(f"   Area inference resolution: {self.yolo.area_inference_size}")
            print(f"   Position mapping method: {self.yolo.position_mapping_method}")
            if self.yolo.shelf_model:
                print(f"   Shelf classes: {list(self.yolo.shelf_model.names.values())}")
            if self.yolo.area_model:
                print(f"   Area classes: {list(self.yolo.area_model.names.values())}")
        except Exception as e:
            print(f"   YOLO test failed: {e}")

        # 3. UART module test
        print("\n3. UART Communication Module Test:")
        try:
            uart_config = self.uart.config['uart']
            print(f"   Configured port: {uart_config['port']}")
            print(f"   Baud rate: {uart_config['baudrate']}")
            print(f"   Signal control: {'Enabled' if self.uart.enable_signal_control else 'Disabled'}")
            print(f"   Pause control: {'Enabled' if self.uart.enable_pause_control else 'Disabled'}")
            print(f"   Serial port connection status: {'Connected' if self.uart.is_connected else 'Not connected'}")

            # Test sending
            test_message = "TEST:SYSTEM"
            send_success = self.uart.send_message(test_message)
            print(f"   Test send: {'Success' if send_success else 'Failure'}")

            # Display signal control information
            if self.uart.enable_signal_control:
                print(f"   Start signal: '{self.uart.start_signal}'")
                if self.uart.enable_pause_control:
                    print(f"   Pause signal: '{self.uart.pause_signal}'")
                    print(f"   Resume signal: '{self.uart.resume_signal}'")
                print(f"   Stop signal: '{self.uart.stop_signal}'")
                print(f"   Request last result signal: '{self.uart.request_last_result_signal}'")
                print(f"   Request perfect result signal: '{self.uart.request_perfect_result_signal}'")

        except Exception as e:
            print(f"   UART test failed: {e}")

        # 4. Signal control function test
        self.test_signal_control()

        # 5. Integrated recognition test
        print("\n5. Integrated Recognition Test:")
        try:
            # Note: in test mode, don't wait for start signal
            original_wait_setting = self.wait_for_start_signal
            self.wait_for_start_signal = False

            # If signal control is enabled, temporarily set to running state
            if self.enable_signal_control:
                original_state = self.uart.get_system_state()
                self.uart.set_system_state(SystemState.RUNNING)
                print(f"   Test mode: Temporarily setting system state to RUNNING")

            result = self.single_recognition_with_signal_control()

            # Restore original settings
            self.wait_for_start_signal = original_wait_setting
            if self.enable_signal_control:
                self.uart.set_system_state(original_state)

            if result['success']:
                print(f"   Integrated test successful!")
                print(f"   Recognition result: {result['result_string']}")
                print(f"   Time elapsed: {result['elapsed_time']:.2f} seconds")
                print(f"   Serial send: {'Success' if result.get('send_success') else 'Failure'}")
                
                # Check if result is perfect
                is_perfect = self.is_perfect_result(result['result_string'])
                print(f"   Perfect result: {'YES' if is_perfect else 'No'}")
                
                if 'recognition_result' in result and 'inference_settings' in result['recognition_result']:
                    print(f"   Inference settings: {result['recognition_result']['inference_settings']}")
            elif result.get('waiting'):
                print(f"   Waiting for start signal...")
            elif result.get('paused'):
                print(f"   System is in a paused state")
            else:
                print(f"   Integrated test failed: {result.get('error')}")
        except Exception as e:
            print(f"   Integrated test exception: {e}")

        print(f"\nTest complete!")

    def print_stats(self):
        """Print statistics including perfect results"""
        if self.stats['total_runs'] > 0:
            success_rate = (self.stats['successful_runs'] / self.stats['total_runs']) * 100

            print(f"\nSystem Statistics:")
            print(f"   Total runs: {self.stats['total_runs']}")
            print(f"   Successful runs: {self.stats['successful_runs']}")
            print(f"   Failed runs: {self.stats['failed_runs']}")
            print(f"   Paused runs: {self.stats['paused_runs']}")
            print(f"   Perfect results found: {self.stats['perfect_results_found']}")
            print(f"   Success rate: {success_rate:.1f}%")
            print(f"   Signals processed: {self.stats['signals_processed']}")
            print(f"   Last result: {self.stats['last_result']}")
            print(f"   Last perfect result: {self.last_perfect_result if self.last_perfect_result else 'None'}")
            print(f"   System state: {self.uart.get_system_state().value}")

            if self.stats['start_time']:
                uptime = datetime.now() - self.stats['start_time']
                print(f"   Uptime: {str(uptime).split('.')[0]}")

    def cleanup(self):
        """Clean up system resources"""
        self.logger.info("Cleaning up system resources...")

        try:
            if self.camera:
                self.camera.release_cameras()

            if self.uart:
                self.uart.disconnect()

            if self.display_active:
                cv2.destroyAllWindows()

        except Exception as e:
            self.logger.error(f"Exception during resource cleanup: {e}")

        print("System shutdown complete")
        self.logger.info("System resource cleanup complete")

    def signal_handler(self, signum, frame):
        """Signal handler"""
        self.logger.info(f"Received signal {signum}, preparing to exit...")
        self.running = False

    # ... (keep all other existing methods unchanged)

    def print_help(self):
        """Display help information"""
        help_text = f"""
Intelligent Delivery Robot Vision Recognition System v2.0

Usage:
  python main.py [option]

Options:
  --single        Execute a single recognition test
  --continuous    Start continuous recognition mode (with signal control)
  --visual        Start visual mode (with signal control)
  --debug         Start debug mode
  --test          Execute a full system test
  --help          Display this help message
  --continuous-matching  Start continuous matching mode (consecutive matches)

New Features:
  Signal Control Functionality
    - Supports start/pause/resume/stop signals from STM32
    - Configurable signal strings
    - Flexible signal control switch

  PERFECT Result Request Functionality
    - Send '{self.uart.request_perfect_result_signal if hasattr(self, 'uart') and self.uart else 'PERFECT'}' signal to request latest PERFECT result
    - PERFECT criteria: Shelf (1-6) all digits 1-6, Area (a-f) one empty + five unique digits 1-6
    - Automatically detects and saves perfect results during recognition

  YOLO Inference Resolution
    - Shelf and area can have separately configured inference resolutions
    - Supports various resolutions like 640, 1280, etc.
    - Balance between performance and accuracy

  Improved Position Mapping
    - Adaptive grid position mapping
    - Combination of relative and absolute positioning
    - Greater robustness

Configuration file: config.json

STM32 Signal Control:
  Send the following strings to the Raspberry Pi's serial port to control detection:
  - '{self.uart.start_signal if hasattr(self, 'uart') and self.uart else 'START'}'  : Start detection
  - '{self.uart.pause_signal if hasattr(self, 'uart') and self.uart else 'PAUSE'}'  : Pause detection
  - '{self.uart.resume_signal if hasattr(self, 'uart') and self.uart else 'RESUME'}' : Resume detection
  - '{self.uart.stop_signal if hasattr(self, 'uart') and self.uart else 'STOP'}'   : Stop detection
  - '{self.uart.request_last_result_signal if hasattr(self, 'uart') and self.uart else 'REQUEST'}' : Request last successful result
  - '{self.uart.request_perfect_result_signal if hasattr(self, 'uart') and self.uart else 'PERFECT'}' : Request last PERFECT result

Examples:
  python main.py --continuous  # Continuous recognition with signal control
  python main.py --visual      # Visual mode with signal control
  python main.py --debug       # Debug mode
  python main.py --test        # System test

Notes:
  1. Ensure cameras are connected
  2. Ensure YOLO model files exist
  3. Check serial port connection and permissions
  4. Press Ctrl+C to exit the program safely
  5. Configure inference resolution to balance performance and accuracy
  6. Signal strings are configured in config.json
  7. PERFECT results are automatically detected and saved for later retrieval
        """
        print(help_text)


def main():
    """Main function - supports quiet mode"""
    try:
        import sys
        import os

        # Check for -q or --quiet parameter
        quiet_mode = '-q' in sys.argv or '--quiet' in sys.argv

        # If quiet mode, immediately set global silence
        if quiet_mode:
            import os
            # Set environment variables to disable warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            os.environ['PYTHONWARNINGS'] = 'ignore'

            # Disable all warnings
            import warnings
            warnings.filterwarnings('ignore')

            # Set log level
            import logging
            logging.getLogger().setLevel(logging.ERROR)
            logging.getLogger('VisionSystem').setLevel(logging.ERROR)
            logging.getLogger('YOLOModule').setLevel(logging.ERROR)
            logging.getLogger('CameraModule').setLevel(logging.ERROR)
            logging.getLogger('UARTModule').setLevel(logging.ERROR)

        # Continue with original code...
        args = [arg for arg in sys.argv[1:] if arg not in ['-q', '--quiet']]
        mode = args[0] if args else "--continuous"

        # Display startup information (reduced output in quiet mode)
        if not quiet_mode:
            print("=" * 60)
            print("Intelligent Delivery Robot Vision Recognition System v2.0")
            print("=" * 60)
            print("Initializing...")
        else:
            print("Vision System v2.0 - Quiet Mode")

        # Create system instance
        vision_system = VisionSystemV2()

        # If quiet mode, adjust log level
        if quiet_mode:
            import logging
            # Set higher log level to reduce output
            logging.getLogger().setLevel(logging.WARNING)
            vision_system.logger.setLevel(logging.WARNING)
            # But keep error messages
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            vision_system.logger.addHandler(console_handler)

        # Handle different modes
        if mode == "--help":
            print_help_with_quiet_option()
            return

        elif mode == "--test":
            # Test mode
            if not quiet_mode:
                print("System Test Mode")
            if vision_system.init_modules():
                vision_system.test_all_modules()
            else:
                print("System initialization failed")

        elif mode == "--single":
            # Single recognition
            if not quiet_mode:
                print("Single Recognition Mode")
            if vision_system.init_modules():
                vision_system.running = True
                result = vision_system.single_recognition_with_signal_control()
                if result['success']:
                    print(f"Result: {result['result_string']}")
                    print(f"Time: {result['elapsed_time']:.2f}s")
                    # Check if perfect
                    if vision_system.is_perfect_result(result['result_string']):
                        print("Perfect result detected!")
                elif result.get('waiting'):
                    print("Waiting for start signal...")
                elif result.get('paused'):
                    print("System is paused")
                else:
                    print(f"Failed: {result.get('error')}")
            else:
                print("System initialization failed")

        elif mode == "--continuous":
            # Traditional continuous recognition mode (voting)
            if not quiet_mode:
                print("Continuous Recognition Mode (voting)")
                print("Press Ctrl+C to exit")
            else:
                print("Continuous Mode (voting) - Starting...")

            # Silent initialization for quiet mode
            if quiet_mode:
                import os
                old_stdout = sys.stdout
                try:
                    devnull = open(os.devnull, 'w')
                    sys.stdout = devnull
                    init_success = vision_system.init_modules()
                finally:
                    if 'devnull' in locals():
                        devnull.close()
                    sys.stdout = old_stdout
            else:
                init_success = vision_system.init_modules()

            if init_success:
                # Display configuration information after successful initialization
                print(f"Shelf Model: {os.path.basename(vision_system.yolo.config['yolo']['shelf_model_path'])}, "
                      f"Device: {vision_system.yolo.device}, "
                      f"Resolution: {vision_system.yolo.shelf_inference_size}, "
                      f"Confidence: {vision_system.yolo.config['yolo']['shelf_confidence_threshold']}")
                print(f"Area Model: {os.path.basename(vision_system.yolo.config['yolo']['area_model_path'])}, "
                      f"Device: {vision_system.yolo.device}, "
                      f"Resolution: {vision_system.yolo.area_inference_size}, "
                      f"Confidence: {vision_system.yolo.config['yolo']['area_confidence_threshold']}")
                print(
                    f"UART: {vision_system.uart.config['uart']['port']}@{vision_system.uart.config['uart']['baudrate']} baud, "
                    f"{vision_system.uart.config['uart']['stopbits']} stop bits, "
                    f"Send count: {vision_system.config['system']['result_send_count']}")
                print(f"PERFECT signal: '{vision_system.uart.request_perfect_result_signal}', REQUEST signal: '{vision_system.uart.request_last_result_signal}'")

                vision_system.continuous_mode_with_signal_control(quiet_mode=quiet_mode)
            else:
                print("System initialization failed")

        elif mode == "--continuous-matching":
            if not quiet_mode:
                print("Continuous Matching Mode")
                print("Requires consecutive identical results to confirm")
                print("Press Ctrl+C to exit")
            else:
                print("Continuous Matching - Starting...")
            # Silent initialization
            if quiet_mode:
                import os
                old_stdout = sys.stdout
                try:
                    devnull = open(os.devnull, 'w')
                    sys.stdout = devnull
                    init_success = vision_system.init_modules()
                finally:
                    if 'devnull' in locals():
                        devnull.close()
                    sys.stdout = old_stdout
            else:
                init_success = vision_system.init_modules()

            if init_success:
                # Display configuration information after successful initialization
                print(f"Shelf Model: {os.path.basename(vision_system.yolo.config['yolo']['shelf_model_path'])}, "
                      f"Device: {vision_system.yolo.device}, "
                      f"Resolution: {vision_system.yolo.shelf_inference_size}, "
                      f"Confidence: {vision_system.yolo.config['yolo']['shelf_confidence_threshold']}")
                print(f"Area Model: {os.path.basename(vision_system.yolo.config['yolo']['area_model_path'])}, "
                      f"Device: {vision_system.yolo.device}, "
                      f"Resolution: {vision_system.yolo.area_inference_size}, "
                      f"Confidence: {vision_system.yolo.config['yolo']['area_confidence_threshold']}")
                print(
                    f"UART: {vision_system.uart.config['uart']['port']}@{vision_system.uart.config['uart']['baudrate']} baud, "
                    f"{vision_system.uart.config['uart']['stopbits']} stop bits, "
                    f"Send count: {vision_system.config['system']['result_send_count']}")
                print(f"PERFECT signal: '{vision_system.uart.request_perfect_result_signal}', REQUEST signal: '{vision_system.uart.request_last_result_signal}'")

                vision_system.continuous_matching_mode(quiet_mode=quiet_mode)
            else:
                print("System initialization failed")

        elif mode == "--visual":
            # Visual mode (doesn't support quiet)
            if quiet_mode:
                print("Note: Visual mode doesn't support quiet mode")
            print("Visual Mode (with signal control)")
            print("Press ESC to quit, SPACE to trigger recognition")

            if vision_system.init_modules():
                vision_system.visual_mode_with_signal_control()
            else:
                print("System initialization failed")

        elif mode == "--debug":
            # Debug mode (doesn't support quiet)
            if quiet_mode:
                print("Note: Debug mode doesn't support quiet mode")
            print("Debug Mode")

            if vision_system.init_modules():
                vision_system.debug_mode()
            else:
                print("System initialization failed")

        else:
            print(f"Unknown option: {mode}")
            print_help_with_quiet_option()

    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print(f"Error: {e}")
        if not quiet_mode:
            import traceback
            traceback.print_exc()
    finally:
        # Ensure resource cleanup
        try:
            if 'vision_system' in locals():
                vision_system.cleanup()
        except:
            pass
        if not quiet_mode:
            print("Program exited")


def print_help_with_quiet_option():
    """Display help information (including quiet option and PERFECT functionality)"""
    help_text = """
Intelligent Delivery Robot Vision Recognition System v2.0

Usage:
  python main.py [options] [-q|--quiet]

Options:
  --single              Execute a single recognition test
  --continuous          Start continuous recognition mode (voting)
  --continuous-matching Start continuous matching mode (consecutive matches)
  --visual              Start visual mode (with signal control)
  --debug               Start debug mode
  --test                Execute a full system test
  --help                Display this help message

Modifiers:
  -q, --quiet           Reduce output verbosity (quiet mode)
                        Works with: --single, --continuous, --continuous-matching

Examples:
  python main.py --continuous                     # Normal output
  python main.py --continuous -q                  # Quiet mode
  python main.py --continuous-matching            # Normal output
  python main.py --continuous-matching -q         # Quiet mode

Continuous Matching Mode:
  - Captures single image per camera each time
  - Requires N consecutive identical results to confirm
  - Configure in config.json under yolo.continuous_matching
  - In quiet mode, only shows essential information:
    * Recognition result string
    * Execution time
    * Confirmation messages

REQUEST Signal Functionality (Both modes):
  - Send 'REQUEST' signal to get last successful result immediately
  - Available in both --continuous and --continuous-matching modes
  - Configured in config.json: "request_last_result_signal": "REQUEST"

PERFECT Signal Functionality (NEW):
  - Send 'PERFECT' signal to get last PERFECT result immediately
  - PERFECT criteria:
    * Shelf (1-6): Must contain all digits 1-6 exactly once, no 'x'
    * Area (a-f): Must contain exactly one empty (0 or x) and five unique digits 1-6
  - Automatically detects and saves perfect results during recognition
  - Available in all recognition modes
  - Configured in config.json: "perfect_result_signal": "PERFECT"
  - In quiet mode, outputs "完美数据" when PERFECT signal is processed

Signal Testing:
  echo "REQUEST" | sudo tee /dev/ttyAMA0    # Request last result
  echo "PERFECT" | sudo tee /dev/ttyAMA0    # Request perfect result
  echo "START" | sudo tee /dev/ttyAMA0      # Start recognition
  echo "STOP" | sudo tee /dev/ttyAMA0       # Stop recognition

Configuration (config.json):
  continuous_matching: {
    "required_matches": 2,      // Number of consecutive matches needed
    "check_interval": 0.2,       // Interval between checks (seconds)
    "send_every_match": true     // Send on every match after confirmation
  }
  
  uart: {
    "request_last_result_signal": "REQUEST",    // Signal for last result
    "perfect_result_signal": "PERFECT"          // Signal for perfect result
  }

Notes:
  - Quiet mode significantly reduces console output
  - Ideal for production environments or long-running tests
  - Errors are still displayed even in quiet mode
  - REQUEST and PERFECT signals work in all recognition modes
  - Perfect results are automatically detected and saved
"""
    print(help_text)


if __name__ == "__main__":
    main()