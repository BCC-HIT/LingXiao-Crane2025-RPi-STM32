#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UART Serial Communication Module - Intelligent Delivery Robot Vision Recognition System
New Features:
1. Signal control functionality (start/pause/resume/stop)
2. Configurable signal strings
3. Asynchronous signal listening
4. Enhanced error handling and reconnection mechanism
5. Complete PERFECT result request functionality
"""

import serial
import time
import json
import os
import threading
import queue
from typing import Dict, Optional, Callable
from datetime import datetime
from enum import Enum


class SystemState(Enum):
    """System state enumeration"""
    STOPPED = "stopped"
    WAITING_START = "waiting_start"
    RUNNING = "running"
    PAUSED = "paused"


class UARTModule:
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize UART serial communication module

        Args:
            config_path: Configuration file path
        """
        self.config = self.load_config(config_path)
        self.serial_port = None
        self.is_connected = False

        # Signal control related
        self.enable_signal_control = self.config['uart'].get('enable_signal_control', False)
        self.enable_pause_control = self.config['uart'].get('enable_pause_control', False)
        self.system_state = SystemState.STOPPED

        # Signal string configuration
        self.start_signal = self.config['uart'].get('start_signal', 'START_DETECTION')
        self.pause_signal = self.config['uart'].get('pause_signal', 'PAUSE_DETECTION')
        self.resume_signal = self.config['uart'].get('resume_signal', 'RESUME_DETECTION')
        self.stop_signal = self.config['uart'].get('stop_signal', 'STOP_DETECTION')
        self.request_last_result_signal = self.config['uart'].get('request_last_result_signal', 'GET_LAST_RESULT')
        self.request_perfect_result_signal = self.config['uart'].get('perfect_result_signal', 'PERFECT')

        # Signal listening related
        self.signal_listener_thread = None
        self.signal_running = False
        self.signal_queue = queue.Queue()
        self.signal_callback = None
        self.request_last_result_callback = None
        self.request_perfect_result_callback = None  # Callback for PERFECT signal
        self.signal_check_interval = self.config['uart'].get('signal_check_interval', 0.1)

        # Repeat sending control
        self.repeat_thread = None
        self.repeat_running = False

        # Statistics
        self.stats = {
            'total_sends': 0,
            'successful_sends': 0,
            'failed_sends': 0,
            'signals_received': 0,
            'last_send_time': None,
            'last_message': '',
            'last_signal': '',
            'last_signal_time': None
        }

        print(f"UART Module initialized successfully")
        print(f"   Signal Control: {'Enabled' if self.enable_signal_control else 'Disabled'}")
        print(f"   Pause Control: {'Enabled' if self.enable_pause_control else 'Disabled'}")
        if self.enable_signal_control:
            print(f"   Start Signal: '{self.start_signal}'")
            print(f"   Pause Signal: '{self.pause_signal}'")
            print(f"   Resume Signal: '{self.resume_signal}'")
            print(f"   Stop Signal: '{self.stop_signal}'")
            print(f"   Request Last Result Signal: '{self.request_last_result_signal}'")
            print(f"   Request Perfect Result Signal: '{self.request_perfect_result_signal}'")

    def load_config(self, config_path: str) -> Dict:
        """Load configuration file"""
        default_config = {
            "uart": {
                "port": "/dev/ttyAMA1",
                "baudrate": 9600,
                "stopbits": 2,
                "timeout": 1,
                "retry_count": 3,
                "retry_delay": 0.5,
                "message_end": "\n",
                "enable_signal_control": False,
                "enable_pause_control": False,
                "start_signal": "START_DETECTION",
                "pause_signal": "PAUSE_DETECTION",
                "resume_signal": "RESUME_DETECTION",
                "stop_signal": "STOP_DETECTION",
                "request_last_result_signal": "REQUEST",
                "perfect_result_signal": "PERFECT",
                "signal_check_interval": 0.1
            }
        }

        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # Merge default configuration
                    if 'uart' not in config:
                        config['uart'] = default_config['uart']
                    else:
                        for key, value in default_config['uart'].items():
                            if key not in config['uart']:
                                config['uart'][key] = value
                    return config
            else:
                return default_config
        except Exception as e:
            print(f"Failed to load configuration file, using default settings: {e}")
            return default_config

    def connect(self) -> bool:
        """Connect to serial port"""
        if self.is_connected:
            return True

        uart_config = self.config['uart']

        try:
            self.serial_port = serial.Serial(
                port=uart_config['port'],
                baudrate=uart_config['baudrate'],
                timeout=uart_config['timeout'],
                stopbits=uart_config['stopbits']
            )

            if self.serial_port.is_open:
                self.is_connected = True
                print(f"Serial port connected successfully: {uart_config['port']}@{uart_config['baudrate']}")

                # If signal control is enabled, start listening
                if self.enable_signal_control:
                    self.start_signal_listener()

                return True
            else:
                print(f"Failed to connect to serial port: {uart_config['port']}")
                return False

        except Exception as e:
            print(f"Serial port connection exception: {e}")
            return False

    def disconnect(self):
        """Disconnect serial port"""
        # Stop signal listening
        self.stop_signal_listener()

        # Stop repeat sending
        self.stop_repeat_send()

        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.is_connected = False

    def start_signal_listener(self):
        """Start signal listening thread"""
        if not self.enable_signal_control or self.signal_running:
            return

        self.signal_running = True

        def signal_worker():
            print(f"Starting to listen for serial signals...")
            while self.signal_running and self.is_connected:
                try:
                    if self.serial_port and self.serial_port.in_waiting > 0:
                        # Read serial data
                        line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                        if line:
                            self.process_received_signal(line)

                    time.sleep(self.signal_check_interval)

                except Exception as e:
                    print(f"Signal listener exception: {e}")
                    time.sleep(1)

        self.signal_listener_thread = threading.Thread(target=signal_worker, daemon=True)
        self.signal_listener_thread.start()

    def stop_signal_listener(self):
        """Stop signal listening"""
        if self.signal_running:
            self.signal_running = False
            if self.signal_listener_thread:
                self.signal_listener_thread.join(timeout=2)

    def process_received_signal(self, signal: str):
        """
        Process received signal

        Args:
            signal: Received signal string
        """
        print(f"Received signal: '{signal}'")

        # Update statistics
        self.stats['signals_received'] += 1
        self.stats['last_signal'] = signal
        self.stats['last_signal_time'] = datetime.now()

        # Process different types of signals
        old_state = self.system_state

        # Handle REQUEST signal for last result
        if signal == self.request_last_result_signal:
            print(f"Received request for last result: '{signal}'")
            if self.request_last_result_callback:
                try:
                    # Call callback directly, don't change main system state
                    self.request_last_result_callback()
                except Exception as e:
                    print(f"Request last result callback exception: {e}")
            return  # Return directly after processing

        # Handle PERFECT signal for perfect result
        if signal == self.request_perfect_result_signal:
            print(f"Received request for perfect result: '{signal}'")
            if self.request_perfect_result_callback:
                try:
                    # Call callback directly, don't change main system state
                    self.request_perfect_result_callback()
                except Exception as e:
                    print(f"Request perfect result callback exception: {e}")
            return  # Return directly after processing

        # Handle state control signals
        if signal == self.start_signal:
            if self.system_state in [SystemState.STOPPED, SystemState.WAITING_START]:
                self.system_state = SystemState.RUNNING
                print(f"System state: {old_state.value} -> {self.system_state.value}")
            else:
                print(f"Current state {self.system_state.value} cannot respond to start signal")

        elif signal == self.pause_signal and self.enable_pause_control:
            if self.system_state == SystemState.RUNNING:
                self.system_state = SystemState.PAUSED
                print(f"System state: {old_state.value} -> {self.system_state.value}")
            else:
                print(f"Current state {self.system_state.value} cannot respond to pause signal")

        elif signal == self.resume_signal and self.enable_pause_control:
            if self.system_state == SystemState.PAUSED:
                self.system_state = SystemState.RUNNING
                print(f"System state: {old_state.value} -> {self.system_state.value}")
            else:
                print(f"Current state {self.system_state.value} cannot respond to resume signal")

        elif signal == self.stop_signal:
            if self.system_state in [SystemState.RUNNING, SystemState.PAUSED]:
                self.system_state = SystemState.STOPPED
                print(f"System state: {old_state.value} -> {self.system_state.value}")
            else:
                print(f"Current state {self.system_state.value} cannot respond to stop signal")
        else:
            print(f"Unrecognized signal: '{signal}'")

        # Put signal in queue for external processing
        self.signal_queue.put({
            'signal': signal,
            'timestamp': datetime.now(),
            'old_state': old_state,
            'new_state': self.system_state
        })

        # If callback function is set, call it
        if self.signal_callback:
            try:
                self.signal_callback(signal, old_state, self.system_state)
            except Exception as e:
                print(f"Signal callback function exception: {e}")

    def set_signal_callback(self, callback: Callable):
        """
        Set signal callback function

        Args:
            callback: Callback function, receives parameters (signal, old_state, new_state)
        """
        self.signal_callback = callback
        print(f"Signal callback function has been set")

    def set_request_last_result_callback(self, callback: Callable):
        """
        Set callback function for requesting last result

        Args:
            callback: Callback function, no parameters
        """
        self.request_last_result_callback = callback
        print(f"Request last result callback function has been set")

    def set_request_perfect_result_callback(self, callback: Callable):
        """
        Set callback function for requesting the latest PERFECT result.
        
        Args:
            callback: Callback function, no parameters
        """
        self.request_perfect_result_callback = callback
        print("Request PERFECT result callback function has been set")

    def get_signal(self, timeout: Optional[float] = None) -> Optional[Dict]:
        """
        Get next signal (blocking)

        Args:
            timeout: Timeout duration (seconds), None for infinite wait

        Returns:
            Signal dictionary or None (timeout)
        """
        try:
            return self.signal_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def has_signal(self) -> bool:
        """Check if there are pending signals"""
        return not self.signal_queue.empty()

    def get_system_state(self) -> SystemState:
        """Get current system state"""
        return self.system_state

    def set_system_state(self, state: SystemState):
        """Set system state (manual control)"""
        old_state = self.system_state
        self.system_state = state
        print(f"Manually set system state: {old_state.value} -> {state.value}")

    def can_start_detection(self) -> bool:
        """Check if detection can be started"""
        if not self.enable_signal_control:
            return True  # Always can start when signal control is not enabled

        return self.system_state == SystemState.RUNNING

    def should_pause_detection(self) -> bool:
        """Check if detection should be paused"""
        if not self.enable_pause_control:
            return False  # Don't pause when pause control is not enabled

        return self.system_state == SystemState.PAUSED

    def wait_for_start_signal(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for start signal

        Args:
            timeout: Timeout duration (seconds)

        Returns:
            Whether start signal was received
        """
        if not self.enable_signal_control:
            return True  # Return immediately when signal control is not enabled

        self.system_state = SystemState.WAITING_START
        print(f"Waiting for start signal '{self.start_signal}'...")

        start_time = time.time()

        while self.system_state == SystemState.WAITING_START:
            if timeout and (time.time() - start_time) > timeout:
                print(f"Waiting for start signal timed out ({timeout} seconds)")
                return False

            time.sleep(0.1)

        return self.system_state == SystemState.RUNNING

    def send_message(self, message: str) -> bool:
        """Send message to serial port"""
        if not self.is_connected:
            if not self.connect():
                return False

        uart_config = self.config['uart']
        self.stats['total_sends'] += 1

        try:
            # Construct complete message
            full_message = message + uart_config['message_end']

            # Send message
            self.serial_port.write(full_message.encode('utf-8'))
            self.serial_port.flush()

            print(f"Sent: {message}")

            # Update statistics
            self.stats['successful_sends'] += 1
            self.stats['last_send_time'] = datetime.now()
            self.stats['last_message'] = message

            return True

        except Exception as e:
            print(f"Send failed: {e}")
            self.stats['failed_sends'] += 1
            # Connection might be broken, reset state
            self.is_connected = False
            return False

    def send_result(self, result_string: str) -> bool:
        """Send recognition result (with retry mechanism)"""
        uart_config = self.config['uart']
        retry_count = uart_config['retry_count']
        retry_delay = uart_config['retry_delay']

        for attempt in range(retry_count):
            if attempt > 0:
                print(f"Retrying send (Attempt {attempt + 1})...")
                time.sleep(retry_delay)

            if self.send_message(result_string):
                return True

        print(f"Send failed after {retry_count} retries")
        return False

    def start_repeat_send(self, message: str, interval: float = 1.0):
        """Start repeatedly sending specified message"""
        if self.repeat_running:
            print("Repeat send is already running")
            return

        self.repeat_running = True

        def repeat_worker():
            print(f"Starting repeat send: '{message}' (Interval: {interval} seconds)")
            while self.repeat_running:
                if self.send_message(message):
                    pass  # Send successful, continue
                else:
                    print("Error during repeat send, attempting to reconnect...")
                    time.sleep(1)  # Wait a bit when error occurs

                # Wait interval
                for _ in range(int(interval * 10)):  # Split into small segments for easy interruption
                    if not self.repeat_running:
                        break
                    time.sleep(0.1)

            print("Repeat send stopped")

        self.repeat_thread = threading.Thread(target=repeat_worker, daemon=True)
        self.repeat_thread.start()

    def stop_repeat_send(self):
        """Stop repeat sending"""
        if self.repeat_running:
            self.repeat_running = False
            if self.repeat_thread:
                self.repeat_thread.join(timeout=2)
            print("Repeat send has been stopped")

    def get_stats(self) -> Dict:
        """Get statistics"""
        success_rate = 0
        if self.stats['total_sends'] > 0:
            success_rate = (self.stats['successful_sends'] / self.stats['total_sends']) * 100

        return {
            **self.stats,
            'success_rate': success_rate,
            'is_connected': self.is_connected,
            'system_state': self.system_state.value,
            'signal_control_enabled': self.enable_signal_control,
            'pause_control_enabled': self.enable_pause_control
        }

    def print_signal_help(self):
        """Print signal control help information"""
        print("\n" + "=" * 60)
        print("UART Signal Control Description")
        print("=" * 60)
        print(f"Signal Control Function: {'Enabled' if self.enable_signal_control else 'Disabled'}")
        print(f"Pause Control Function: {'Enabled' if self.enable_pause_control else 'Disabled'}")

        if self.enable_signal_control:
            print(f"\nControl signals to be sent by STM32:")
            print(f"  Start Detection: '{self.start_signal}'")
            if self.enable_pause_control:
                print(f"  Pause Detection: '{self.pause_signal}'")
                print(f"  Resume Detection: '{self.resume_signal}'")
            print(f"  Stop Detection: '{self.stop_signal}'")
            print(f"  Request Last Result: '{self.request_last_result_signal}'")
            print(f"  Request Perfect Result: '{self.request_perfect_result_signal}'")

            print(f"\nSignal Usage Example (sent from STM32):")
            print(f"  1. Send '{self.start_signal}' - Start detection")
            if self.enable_pause_control:
                print(f"  2. Send '{self.pause_signal}' - Pause detection (after current cycle)")
                print(f"  3. Send '{self.resume_signal}' - Resume detection")
            print(f"  4. Send '{self.stop_signal}' - Stop detection")
            print(f"  5. Send '{self.request_last_result_signal}' - Request the latest successful result")
            print(f"  6. Send '{self.request_perfect_result_signal}' - Request the latest PERFECT result")

            print(f"\nSystem State Transitions:")
            print(f"  STOPPED -> RUNNING: on receiving '{self.start_signal}'")
            if self.enable_pause_control:
                print(f"  RUNNING -> PAUSED: on receiving '{self.pause_signal}'")
                print(f"  PAUSED -> RUNNING: on receiving '{self.resume_signal}'")
            print(f"  Any State -> STOPPED: on receiving '{self.stop_signal}'")
            
            print(f"\nSpecial Request Signals:")
            print(f"  '{self.request_last_result_signal}': Returns last successful recognition result immediately")
            print(f"  '{self.request_perfect_result_signal}': Returns last PERFECT recognition result (if any)")
            print(f"    - PERFECT result criteria:")
            print(f"      * Shelf (1-6): All digits 1-6, no duplicates, no 'x'")
            print(f"      * Area (a-f): Exactly one empty position (0 or x), other 5 positions are unique digits 1-6")
        else:
            print(f"\nTo enable signal control, modify the configuration file:")
            print(f'  "enable_signal_control": true')
            print(f'  "enable_pause_control": true')

        print(f"\nCurrent State: {self.system_state.value}")
        print("=" * 60)

    def test_communication(self):
        """Test serial communication"""
        print("=" * 50)
        print("UART Serial Communication Module Test")
        print("=" * 50)

        uart_config = self.config['uart']
        print(f"Serial Port Configuration: {uart_config['port']}")
        print(f"   Baud rate: {uart_config['baudrate']}")
        print(f"   Stop Bits: {uart_config.get('stopbits', 1)}")

        # Display signal control configuration
        self.print_signal_help()

        # 1. Connection test
        print(f"\n1. Connection Test:")
        if self.connect():
            print("   Connection successful")

            # 2. Signal control test
            if self.enable_signal_control:
                print(f"\n2. Signal Control Test:")
                print(f"   Please send signals from STM32 for testing...")
                print(f"   Waiting 5 seconds to observe signal reception...")

                for i in range(50):  # Wait 5 seconds
                    if self.has_signal():
                        signal_data = self.get_signal(0.1)
                        if signal_data:
                            print(f"   Signal received: {signal_data}")
                    time.sleep(0.1)

            # 3. Send test
            print(f"\n3. Send Test:")
            test_messages = [
                "TEST:UART",
                "1:2,2:3,3:x,4:1,5:5,6:6,a:x,b:0,c:x,d:x,e:x,f:x",
                "STATUS:READY"
            ]

            for i, msg in enumerate(test_messages, 1):
                print(f"   Test message {i}: {msg}")
                success = self.send_message(msg)
                print(f"   Result: {'Success' if success else 'Failure'}")
                time.sleep(0.5)

            # 4. State test
            print(f"\n4. State Test:")
            print(f"   Current system state: {self.system_state.value}")
            print(f"   Can start detection: {'Yes' if self.can_start_detection() else 'No'}")
            print(f"   Should pause detection: {'Yes' if self.should_pause_detection() else 'No'}")

            # 5. Statistics
            print(f"\n5. Statistics:")
            stats = self.get_stats()
            print(f"   Total sends: {stats['total_sends']}")
            print(f"   Successful sends: {stats['successful_sends']}")
            print(f"   Failed sends: {stats['failed_sends']}")
            print(f"   Success rate: {stats['success_rate']:.1f}%")
            print(f"   Signals received: {stats['signals_received']}")
            print(f"   Last signal: {stats['last_signal']}")

            # 6. Disconnect
            self.disconnect()

        else:
            print("   Connection failed")
            print(f"\nTroubleshooting Suggestions:")
            print(f"   1. Check if the serial device is connected: {uart_config['port']}")
            print(f"   2. Check device permissions: sudo chmod 666 {uart_config['port']}")
            print(f"   3. Check if other programs are using the serial port")
            print(f"   4. Modify the serial port path in the configuration file")
            print(f"   5. Send '{self.request_last_result_signal}' - Request latest successful result")
            print(f"   6. Send '{self.request_perfect_result_signal}' - Request latest PERFECT result")

        print(f"\nTest complete!")


def main():
    """Main function - test entry point"""
    try:
        import sys

        # Create UART module instance
        uart_module = UARTModule()

        if len(sys.argv) > 1:
            mode = sys.argv[1]

            if mode == "--send":
                # Send specified message
                if len(sys.argv) > 2:
                    message = sys.argv[2]
                    success = uart_module.send_result(message)
                    print(f"Send result: {'Success' if success else 'Failure'}")
                    uart_module.disconnect()
                else:
                    print("Usage: python uart_module.py --send <message>")

            elif mode == "--listen":
                # Signal listening mode
                print(f"Signal Listener Mode")
                uart_module.print_signal_help()

                if uart_module.connect():
                    print(f"\nStarting to listen for signals, press Ctrl+C to stop...")
                    try:
                        while True:
                            signal_data = uart_module.get_signal(1.0)
                            if signal_data:
                                print(f"Received signal: {signal_data}")
                    except KeyboardInterrupt:
                        print("\nUser interrupted listener")
                    finally:
                        uart_module.disconnect()

            elif mode == "--help":
                # Display help
                uart_module.print_signal_help()

            else:
                print(f"Unknown parameter: {mode}")
                print("Available parameters:")
                print("  --send <message>     # Send a single message")
                print("  --listen          # Signal listener mode")
                print("  --help            # Display signal control help")
        else:
            # Default run complete test
            uart_module.test_communication()

    except Exception as e:
        print(f"Program runtime error: {e}")


if __name__ == "__main__":
    main()