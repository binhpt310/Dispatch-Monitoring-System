#!/usr/bin/env python3
"""
Video Streaming Application Launcher
Professional launcher script for the refactored video streaming application
"""

import os
import sys
import threading
import time
import signal
import atexit
import requests
from pathlib import Path

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent / "video_streaming_app"))

from video_streaming_app.core.config import Config
from video_streaming_app.core.video_processor import VideoProcessor
from video_streaming_app.core.feedback_manager import FeedbackManager
from video_streaming_app.routes.api_routes import configure_routes
from flask import Flask, request


def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        from flask import Flask
        import requests
        print("-  All dependencies are available")
        return True
    except ImportError as e:
        print(f"-  Missing dependency: {e}")
        print("-  Please install required packages: pip install -r requirements.txt")
        return False

def check_files():
    """Check if all required files exist"""
    required_files = [
        Config.DETECTION_MODEL_PATH,
        Config.CLASSIFICATION_MODEL_PATH,
        Config.VIDEO_PATH
    ]
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    if missing_files:
        print("-  Missing required files:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        return False
    print("-  All required files are present")
    return True

class VideoStreamingApp:
    """Main application class for the video streaming system"""
    def __init__(self):
        print("-  Initializing Video Streaming Application...")
        self.video_processor = None
        self.feedback_manager = None
        self.flask_app = None
        self.video_thread = None
        self.flask_thread = None
        self.running = False
        self.shutdown_requested = False
        self.shutdown_called = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.shutdown)
        
    def _signal_handler(self, signum, frame):
        if self.shutdown_called:
            print(f"\n🔴 Force exit requested...")
            os._exit(1)
        print(f"\n-  Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_requested = True
        self.shutdown()
        sys.exit(0)
    def initialize_components(self):
        try:
            print("-  Initializing application components...")
            print("-  Initializing feedback manager...")
            self.feedback_manager = FeedbackManager()
            print("🎥 Initializing video processor...")
            self.video_processor = VideoProcessor()
            print("🤖 Loading AI models...")
            if not self.video_processor.load_models():
                print("-  Failed to load models")
                return False
            print("📹 Loading video file...")
            if not self.video_processor.load_video():
                print("-  Failed to load video")
                return False
            print("-  Initializing Flask web server...")
            # Set correct path for templates
            app_dir = Path(__file__).parent / "video_streaming_app"
            template_dir = app_dir / "templates"
            
            self.flask_app = Flask(
                __name__, 
                template_folder=str(template_dir)
            )
            
            # Add shutdown route
            @self.flask_app.route('/shutdown', methods=['POST'])
            def shutdown_server():
                func = request.environ.get('werkzeug.server.shutdown')
                if func is not None:
                    func()
                    return 'Server shutting down...'
                else:
                    # Alternative shutdown method
                    os._exit(0)
            
            configure_routes(self.flask_app, self.video_processor, self.feedback_manager)
            print("-  All components initialized successfully!")
            return True
        except Exception as e:
            print(f"-  Error initializing components: {e}")
            return False
    def start_video_processing(self):
        try:
            print("🎬 Starting video processing thread...")
            self.video_processor.running = True
            self.video_thread = threading.Thread(
                target=self._video_processing_loop,
                daemon=True,
                name="VideoProcessingThread"
            )
            self.video_thread.start()
            print("-  Video processing started successfully!")
            return True
        except Exception as e:
            print(f"-  Error starting video processing: {e}")
            return False
    def _video_processing_loop(self):
        print("-  Video processing loop started")
        while self.video_processor.running and not self.shutdown_requested:
            try:
                time.sleep(0.1)
            except Exception as e:
                print(f"-  Video processing loop error: {e}")
                time.sleep(1)
        print("-  Video processing loop ended")
    def _flask_server_thread(self):
        """Run Flask server in a separate thread"""
        try:
            self.flask_app.run(
                host=Config.HOST,
                port=Config.PORT,
                debug=Config.DEBUG,
                threaded=Config.THREADED,
                use_reloader=False
            )
        except Exception as e:
            if not self.shutdown_requested:
                print(f"-  Error in Flask server: {e}")
    
    def start_web_server(self):
        try:
            print(f"-  Starting web server on {Config.HOST}:{Config.PORT}...")
            print(f"📡 Access the application at: http://{Config.HOST}:{Config.PORT}")
            
            # Start Flask server in a separate thread
            self.flask_thread = threading.Thread(
                target=self._flask_server_thread,
                daemon=True,
                name="FlaskServerThread"
            )
            self.flask_thread.start()
            
            # Keep main thread alive and responsive to signals
            while self.running and not self.shutdown_requested:
                time.sleep(0.1)
                
        except Exception as e:
            print(f"-  Error starting web server: {e}")
            return False
    def run(self):
        try:
            print("=" * 60)
            print("🎬 VIDEO STREAMING APPLICATION")
            print("=" * 60)
            if not self.initialize_components():
                print("-  Failed to initialize application")
                return False
            if not self.start_video_processing():
                print("-  Failed to start video processing")
                return False
            self.running = True
            print("🎉 Application started successfully!")
            print("-" * 60)
            print("-  SYSTEM STATUS:")
            print(f"   • Video: {self.video_processor.video_path}")
            print(f"   • Detection Model: {self.video_processor.detection_model_path}")
            print(f"   • Classification Model: {self.video_processor.classification_model_path}")
            print(f"   • Feedback Database: {self.feedback_manager.db_path}")
            print(f"   • Web Server: http://{Config.HOST}:{Config.PORT}")
            print("-" * 60)
            print("⚡ Ready to stream! Press Ctrl+C to stop.")
            print("=" * 60)
            self.start_web_server()
        except KeyboardInterrupt:
            print("\n-  Keyboard interrupt received")
        except Exception as e:
            print(f"-  Application error: {e}")
        finally:
            self.shutdown()
    def shutdown(self):
        if self.shutdown_called:
            return
        self.shutdown_called = True
        
        print("\n-  Shutting down application...")
        self.running = False
        self.shutdown_requested = True
        
        try:
            # Stop video processor
            if self.video_processor:
                print("🎥 Stopping video processor...")
                self.video_processor.running = False
                self.video_processor.cleanup()
            
            # Wait for video thread to finish
            if self.video_thread and self.video_thread.is_alive():
                print("-  Waiting for video thread to finish...")
                self.video_thread.join(timeout=3)
            
            # Shutdown Flask server
            if self.flask_thread and self.flask_thread.is_alive():
                print("-  Stopping web server...")
                try:
                    requests.post(f'http://localhost:{Config.PORT}/shutdown', timeout=2)
                except:
                    pass  # Server might already be down
                
            print("-  Application shutdown completed")
        except Exception as e:
            print(f"-  Error during shutdown: {e}")
        finally:
            # Ensure we exit
            print("👋 Goodbye!")
            time.sleep(0.5)  # Give a moment for cleanup
            os._exit(0)

def main():
    print("-  Checking system requirements...")
    if not check_dependencies():
        sys.exit(1)
    if not check_files():
        sys.exit(1)
    app = VideoStreamingApp()
    app.run()

if __name__ == "__main__":
    main() 