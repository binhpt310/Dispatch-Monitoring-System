#!/usr/bin/env python3
"""
Video Processing Module
Handles video capture, frame processing, detection, and classification
"""

import cv2
import numpy as np
import time
import threading
import torch
from pathlib import Path
from ultralytics import YOLO

from .config import Config


class VideoProcessor:
    """Video processor with detection and classification capabilities"""
    
    def __init__(self):
        """Initialize video processor with configuration"""
        # Device detection and configuration
        self.device = self._detect_device()
        print(f"- Using device: {self.device}")
        
        # Model paths
        self.detection_model_path = Config.DETECTION_MODEL_PATH
        self.classification_model_path = Config.CLASSIFICATION_MODEL_PATH
        
        # Models
        self.detection_model = None
        self.classification_model = None
        
        # Video processing
        self.video_path = Config.VIDEO_PATH
        self.cap = None
        self.cap_lock = threading.Lock()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        self.processing_times = []
        
        # Configuration
        self.class_names = Config.CLASS_NAMES
        self.class_colors = Config.CLASS_COLORS
        self.confidence_threshold = Config.CONFIDENCE_THRESHOLD
        self.inference_area = Config.INFERENCE_AREA
        self.skip_frames = Config.SKIP_FRAMES
        
        # Processing settings
        self.running = False
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.frame_count = 0
        
        # Video playback controls
        self.is_playing = True
        self.current_frame_number = 0
        self.total_frames = 0
        self.video_fps = 30
        self.target_frame = None
        
        # Class counting
        self.current_detections = []
        self.class_counts = {
            'dish_empty': 0,
            'dish_kakigori': 0, 
            'dish_not_empty': 0,
            'tray_empty': 0,
            'tray_kakigori': 0,
            'tray_not_empty': 0
        }
    
    def _detect_device(self):
        """Detect available device (CUDA/CPU) and return appropriate device string"""
        try:
            # Check for forced CPU mode via environment variable
            import os
            if os.getenv('FORCE_CPU', '').lower() in ['true', '1', 'yes']:
                print("- FORCE_CPU enabled - Using CPU mode")
                return 'cpu'
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                print(f"- CUDA available: {gpu_count} GPU(s) - Using {gpu_name}")
                return 'cuda'
            else:
                print("- CUDA not available - Using CPU (slower inference)")
                return 'cpu'
        except Exception as e:
            print(f"- Device detection error: {e} - Falling back to CPU")
            return 'cpu'
    
    def load_models(self):
        """Load detection and classification models with CPU/GPU support"""
        try:
            print("- Loading models...")
            
            # Load detection model
            if Path(self.detection_model_path).exists():
                self.detection_model = YOLO(self.detection_model_path)
                # Move model to appropriate device
                if hasattr(self.detection_model, 'model') and self.detection_model.model:
                    self.detection_model.model = self.detection_model.model.to(self.device)
                print(f"- Detection model loaded: {self.detection_model_path} ({self.device})")
            else:
                print(f"- Detection model not found: {self.detection_model_path}")
                return False
            
            # Load classification model  
            if Path(self.classification_model_path).exists():
                self.classification_model = YOLO(self.classification_model_path)
                # Move model to appropriate device
                if hasattr(self.classification_model, 'model') and self.classification_model.model:
                    self.classification_model.model = self.classification_model.model.to(self.device)
                print(f"- Classification model loaded: {self.classification_model_path} ({self.device})")
            else:
                print(f"- Classification model not found: {self.classification_model_path}")
                return False
            
            return True
            
        except Exception as e:
            print(f"- Error loading models: {e}")
            return False
    
    def load_video(self):
        """Load video with robust error handling"""
        try:
            if not Path(self.video_path).exists():
                print(f"- Video file not found: {self.video_path}")
                return False
            
            # Release any existing capture
            if self.cap:
                self.cap.release()
                time.sleep(0.1)
            
            # Initialize with threading-safe settings
            self.cap = cv2.VideoCapture(self.video_path)
            
            # Set threading-safe properties
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.BUFFER_SIZE)
            
            if not self.cap.isOpened():
                print(f"- Cannot open video: {self.video_path}")
                return False
            
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Store video properties
            self.video_fps = fps if fps > 0 else 30
            self.total_frames = frame_count
            
            print(f"- Video loaded: {width}x{height}, {fps:.1f} FPS, {frame_count} frames")
            return True
            
        except Exception as e:
            print(f"- Error loading video: {e}")
            return False
    
    def process_frame_fast(self, frame):
        """Ultra-fast frame processing with detection and classification - CPU/GPU optimized"""
        try:
            start_time = time.time()
            
            # Get frame dimensions for inference area calculation
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate inference area coordinates
            area_x1 = int(self.inference_area['x1'] * frame_width)
            area_y1 = int(self.inference_area['y1'] * frame_height)
            area_x2 = int(self.inference_area['x2'] * frame_width)
            area_y2 = int(self.inference_area['y2'] * frame_height)
            
            # Extract inference region
            inference_region = frame[area_y1:area_y2, area_x1:area_x2]
            

            imgsz = 640
            half = False  # Can be enabled for newer GPUs
            
            # Fast Detection on inference region only
            detection_results = self.detection_model(
                inference_region,
                conf=self.confidence_threshold,
                verbose=False,
                imgsz=imgsz,
                device=self.device,
                half=False,
                augment=False
            )
            
            # Create output frame
            output_frame = frame.copy()
            detections_info = []
            
            # Process detections
            if detection_results and len(detection_results) > 0:
                detection_result = detection_results[0]
                
                if detection_result.boxes is not None:
                    for box in detection_result.boxes:
                        # Get detection coordinates (relative to inference region)
                        region_x1, region_y1, region_x2, region_y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        # Convert coordinates back to full frame
                        x1 = region_x1 + area_x1
                        y1 = region_y1 + area_y1
                        x2 = region_x2 + area_x1
                        y2 = region_y2 + area_y1
                        
                        detection_conf = float(box.conf[0].cpu().numpy())
                        detection_cls = int(box.cls[0].cpu().numpy())
                        detection_type = self.detection_model.names[detection_cls]
                        
                        # Fast Classification on detected region
                        final_class = detection_type
                        final_conf = detection_conf
                        
                        # Quick crop for classification
                        padding = 5
                        crop_x1 = max(0, x1 - padding)
                        crop_y1 = max(0, y1 - padding)
                        crop_x2 = min(frame.shape[1], x2 + padding)
                        crop_y2 = min(frame.shape[0], y2 + padding)
                        
                        cropped_region = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                        
                        if cropped_region.size > 0:
                            # Fast classification
                            classification_results = self.classification_model(
                                cropped_region,
                                conf=0.1,
                                verbose=False,
                                imgsz=224,
                                device=self.device,
                                half=False,
                                augment=False
                            )
                            
                            if classification_results and len(classification_results) > 0:
                                class_result = classification_results[0]
                                if hasattr(class_result, 'probs') and class_result.probs is not None:
                                    top_class_id = class_result.probs.top1
                                    classification_conf = class_result.probs.top1conf.item()
                                    
                                    if top_class_id in self.class_names:
                                        final_class = self.class_names[top_class_id]
                                        final_conf = (detection_conf * 0.7 + classification_conf * 0.3)
                        
                        # Force classification if still basic
                        if final_class in ['dish', 'tray']:
                            if detection_type == 'dish':
                                final_class = 'dish_not_empty'
                            elif detection_type == 'tray':
                                final_class = 'tray_not_empty'
                        
                        # Drawing with matching colors
                        color = self.class_colors.get(final_class, (0, 255, 0))
                        
                        # Draw bounding box with configurable thickness
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, Config.BBOX_THICKNESS)
                        
                        # Draw label with configurable font settings
                        label = f"{final_class}: {final_conf:.2f}"
                        font_scale = Config.BBOX_FONT_SCALE
                        font_thickness = Config.BBOX_FONT_THICKNESS
                        
                        # Calculate text size
                        (text_width, text_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                        )
                        
                        # Draw text background with matching color and configurable padding
                        cv2.rectangle(
                            output_frame,
                            (x1, y1 - text_height - Config.BBOX_TEXT_PADDING),
                            (x1 + text_width + Config.BBOX_TEXT_OFFSET_X * 2, y1),
                            color,
                            -1
                        )
                        
                        # Draw text with configurable positioning
                        cv2.putText(
                            output_frame,
                            label,
                            (x1 + Config.BBOX_TEXT_OFFSET_X, y1 - Config.BBOX_TEXT_OFFSET_Y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (0, 0, 0),  # Black text for good contrast against colored background
                            font_thickness
                        )
                        
                        detections_info.append({
                            'class': final_class,
                            'confidence': final_conf,
                            'bbox': [x1, y1, x2, y2]
                        })
            
            # Update class counts and current detections
            self.current_detections = detections_info
            self.class_counts = {
                'dish_empty': 0,
                'dish_kakigori': 0, 
                'dish_not_empty': 0,
                'tray_empty': 0,
                'tray_kakigori': 0,
                'tray_not_empty': 0
            }
            
            for detection in detections_info:
                if detection['class'] in self.class_counts:
                    self.class_counts[detection['class']] += 1
            
            # Add performance info
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            if len(self.processing_times) > Config.PROCESSING_TIMES_BUFFER:
                self.processing_times = self.processing_times[-Config.PROCESSING_TIMES_BUFFER:]
            
            avg_processing_time = sum(self.processing_times) / len(self.processing_times)
            
            fps_text = f"FPS: {self.current_fps:.1f} | Proc: {processing_time*1000:.1f}ms | Avg: {avg_processing_time*1000:.1f}ms"
            cv2.putText(
                output_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            # Detection count
            if detections_info:
                count_text = f"Detections: {len(detections_info)}"
                cv2.putText(
                    output_frame,
                    count_text,
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )
            
            return output_frame
            
        except Exception as e:
            print(f"- Frame processing error: {e}")
            return frame
    
    def video_stream_generator(self):
        """Generate video stream with robust error handling"""
        retry_count = 0
        max_retries = Config.MAX_RETRIES
        
        while self.running:
            try:
                # Ensure video capture is properly initialized with thread safety
                with self.cap_lock:
                    if not self.cap or not self.cap.isOpened():
                        print("- Reinitializing video capture...")
                        if self.cap:
                            self.cap.release()
                        time.sleep(0.1)
                        self.cap = cv2.VideoCapture(self.video_path)
                        
                        # Set threading-safe properties
                        if self.cap.isOpened():
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, Config.BUFFER_SIZE)
                        
                        if not self.cap.isOpened():
                            print("- Failed to reopen video capture")
                            time.sleep(1)
                            continue
                
                # Handle seeking with error checking
                if self.target_frame is not None:
                    try:
                        with self.cap_lock:
                            if self.cap and self.cap.isOpened():
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.target_frame)
                                self.current_frame_number = self.target_frame
                        self.target_frame = None
                        retry_count = 0
                    except Exception as e:
                        print(f"- Seek error: {e}")
                        self.target_frame = None
                
                # Handle pause
                if not self.is_playing:
                    time.sleep(0.1)
                    continue
                
                # Read frame with thread safety
                with self.cap_lock:
                    if self.cap and self.cap.isOpened():
                        ret, frame = self.cap.read()
                        if ret:
                            self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    else:
                        ret, frame = False, None
                
                if not ret:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"-  Frame read failed, retry {retry_count}/{max_retries}")
                        time.sleep(0.1)
                        continue
                    else:
                        # Loop video after max retries
                        print("-  Looping video...")
                        with self.cap_lock:
                            if self.cap and self.cap.isOpened():
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                self.current_frame_number = 0
                        retry_count = 0
                        continue
                
                # Reset retry count on successful frame read
                retry_count = 0
                self.frame_count += 1
                
                # Process frame safely
                try:
                    if self.frame_count % self.skip_frames == 0:
                        processed_frame = self.process_frame_fast(frame)
                    else:
                        processed_frame = frame
                except Exception as e:
                    print(f"-  Frame processing error: {e}")
                    processed_frame = frame
                
                # Update FPS
                self.fps_counter += 1
                current_time = time.time()
                if current_time - self.fps_time >= 1.0:
                    self.current_fps = self.fps_counter / (current_time - self.fps_time)
                    self.fps_counter = 0
                    self.fps_time = current_time
                
                # Store current frame
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()
                
                # Encode frame for web streaming
                try:
                    # Resize for web with higher resolution and better quality
                    height, width = processed_frame.shape[:2]
                    if width > Config.MAX_WIDTH:
                        scale = Config.MAX_WIDTH / width
                        new_width = Config.MAX_WIDTH
                        new_height = int(height * scale)
                        processed_frame = cv2.resize(processed_frame, (new_width, new_height), 
                                                   interpolation=cv2.INTER_LANCZOS4)
                    
                    # Encode as JPEG with higher quality
                    ret, buffer = cv2.imencode('.jpg', processed_frame, [
                        cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY,
                        cv2.IMWRITE_JPEG_OPTIMIZE, 1
                    ])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                except Exception as e:
                    print(f"-  Encoding error: {e}")
                
                # Small delay to prevent overload
                time.sleep(Config.FRAME_DELAY)
                
            except Exception as e:
                print(f"-  Video stream error: {e}")
                time.sleep(Config.RETRY_DELAY)
    
    def get_stats(self):
        """Get performance statistics"""
        avg_proc_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        return {
            'fps': self.current_fps,
            'avg_processing_time_ms': avg_proc_time * 1000,
            'frame_count': self.frame_count,
            'models_loaded': self.detection_model is not None and self.classification_model is not None,
            'video_info': {
                'total_frames': self.total_frames,
                'current_frame': self.current_frame_number,
                'video_fps': self.video_fps,
                'is_playing': self.is_playing
            },
            'class_counts': self.class_counts,
            'current_detections': len(self.current_detections)
        }
    
    def toggle_playback(self):
        """Toggle play/pause"""
        self.is_playing = not self.is_playing
        return self.is_playing
    
    def seek_to_frame(self, frame_number):
        """Seek to specific frame"""
        if 0 <= frame_number < self.total_frames:
            self.target_frame = frame_number
            return True
        return False
    
    def cleanup(self):
        """Clean up video resources"""
        print("-  Cleaning up video resources...")
        self.running = False
        
        with self.cap_lock:
            if self.cap:
                self.cap.release()
                self.cap = None
        
        print("-  Video cleanup completed") 