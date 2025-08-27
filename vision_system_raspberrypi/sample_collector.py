# sample_collector_x11.py
#
# A remote camera sample collection tool via X11 forwarding.
# Supports both single camera and dual camera modes.

import cv2
import json
import argparse
import datetime
import os
import numpy as np


def load_camera_config(config_path="config.json"):
    """Load camera configuration"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Successfully loaded configuration file.")
        return config.get('camera', {})
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{config_path}' format error.")
        exit(1)


def set_camera_params(camera, config):
    """Set camera parameters according to configuration file"""
    print("Applying camera parameters...")
    params_to_set = [
        # (config_key, cv2_property)
        ('width', cv2.CAP_PROP_FRAME_WIDTH),
        ('height', cv2.CAP_PROP_FRAME_HEIGHT),
        ('fps', cv2.CAP_PROP_FPS),
        ('brightness', cv2.CAP_PROP_BRIGHTNESS),
        ('contrast', cv2.CAP_PROP_CONTRAST),
        ('saturation', cv2.CAP_PROP_SATURATION),
        ('hue', cv2.CAP_PROP_HUE),
        ('gain', cv2.CAP_PROP_GAIN),
        ('exposure_time_absolute', cv2.CAP_PROP_EXPOSURE),
        ('sharpness', cv2.CAP_PROP_SHARPNESS),
        ('backlight_compensation', cv2.CAP_PROP_BACKLIGHT),
        ('gamma', cv2.CAP_PROP_GAMMA),
    ]

    # Boolean values need special handling
    if "auto_exposure" in config:
        # 0.75 = auto, 0.25 = manual
        val = 0.75 if config["auto_exposure"] else 0.25
        camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
        print(f"   - auto_exposure: {config['auto_exposure']}")

    if "auto_white_balance" in config:
        val = 1 if config["auto_white_balance"] else 0
        camera.set(cv2.CAP_PROP_AUTO_WB, val)
        print(f"   - auto_white_balance: {config['auto_white_balance']}")

    # Set numeric parameters
    for key, prop in params_to_set:
        if key in config:
            value = config[key]
            camera.set(prop, float(value))
            print(f"   - {key}: {value}")


def initialize_camera(device_path, config):
    """Initialize single camera and set parameters"""
    camera = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
    if not camera.isOpened():
        print(f"Error: Unable to open camera {device_path}")
        return None

    print(f"\nInitializing camera: {device_path}")
    set_camera_params(camera, config)
    return camera


def main():
    parser = argparse.ArgumentParser(description="Remote Camera Sample Collection Tool (X11 Mode)")
    parser.add_argument("--device", help="Specify single camera device path to start single camera mode (e.g. /dev/camera_shelf)")
    args = parser.parse_args()

    config = load_camera_config()

    # Preview window dimensions
    preview_width = 480
    preview_height = 270  # 16:9 aspect ratio

    # Create folder for storing samples
    sample_dir = "samples_x11"
    os.makedirs(sample_dir, exist_ok=True)
    print(f"\nPhotos taken will be saved in '{sample_dir}' folder.")

    cameras = {}
    if args.device:
        # --- Single Camera Mode ---
        print("Mode: Single Camera")
        cam = initialize_camera(args.device, config)
        if cam:
            cameras[os.path.basename(args.device)] = cam
    else:
        # --- Dual Camera Mode ---
        print("Mode: Dual Camera")
        shelf_device = config.get("shelf_camera_device", "/dev/camera_shelf")
        area_device = config.get("area_camera_device", "/dev/camera_area")

        cam_shelf = initialize_camera(shelf_device, config)
        cam_area = initialize_camera(area_device, config)

        if cam_shelf:
            cameras['shelf'] = cam_shelf
        if cam_area:
            cameras['area'] = cam_area

    if not cameras:
        print("No cameras started successfully, exiting program.")
        return

    print("\nPreview window will popup... Please operate on the [preview window]:")
    print("   - Press 'c' key to capture (capture)")
    print("   - Press 'q' key to quit (quit)")

    try:
        while True:
            full_res_frames = {}
            preview_frames = {}

            # Read frames from all active cameras
            for name, cam in cameras.items():
                ret, frame = cam.read()
                if ret:
                    full_res_frames[name] = frame
                    preview = cv2.resize(frame, (preview_width, preview_height))
                    # Add text to preview image
                    cv2.putText(preview, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    preview_frames[name] = preview
                else:
                    # If read fails, create a black image as placeholder
                    preview_frames[name] = np.zeros((preview_height, preview_width, 3), dtype=np.uint8)
                    cv2.putText(preview_frames[name], f"{name} ERROR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2)

            # Display preview window
            if len(preview_frames) == 1:
                cv2.imshow("Camera Preview", list(preview_frames.values())[0])
            elif len(preview_frames) == 2:
                # Horizontally combine two preview images into one window
                combined_preview = np.hstack((preview_frames.get('shelf'), preview_frames.get('area')))
                cv2.imshow("Dual Camera Preview (Shelf | Area)", combined_preview)

            # Wait for key press
            key = cv2.waitKey(200) & 0xFF

            if key == ord('q'):
                print("\nReceived quit command, closing...")
                break

            elif key == ord('c'):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
                for name, frame in full_res_frames.items():
                    filename = os.path.join(sample_dir, f"{name}_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Photo saved: {filename}")

    finally:
        # Release all camera resources and close windows
        for cam in cameras.values():
            cam.release()
        cv2.destroyAllWindows()
        print("Program exited safely.")


if __name__ == '__main__':
    main()