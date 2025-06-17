#!/usr/bin/env python3
"""
Configuration file for Video Streaming Application
Contains all application settings and constants
"""

import os
from pathlib import Path

class Config:
    """Application configuration class"""
    
    # Model Paths
    DETECTION_MODEL_PATH = 'models/detection_model_yolo12s.pt'
    CLASSIFICATION_MODEL_PATH = 'models/classification_model_yolo11s-cls_best.pt'
    
    # Video Settings
    VIDEO_PATH = "testing_video.mp4"
    CONFIDENCE_THRESHOLD = 0.5
    SKIP_FRAMES = 1  # Process every frame
    
    # Inference Area (normalized coordinates)
    # INFERENCE_AREA = {
    # 'x1': 0.0,
    # 'y1': 0.0, 
    # 'x2': 1.0,  # Full width
    # 'y2': 1.0   # Full height
    # }

    INFERENCE_AREA = {
        'x1': 0.473,
        'y1': 0.000, 
        'x2': 0.710,
        'y2': 0.249
    }
    
    # Class Names and Colors (BGR format for OpenCV)
    CLASS_NAMES = {
        0: 'dish_empty',
        1: 'dish_kakigori', 
        2: 'dish_not_empty',
        3: 'tray_empty',
        4: 'tray_kakigori',
        5: 'tray_not_empty'
    }
    
    CLASS_COLORS = {
        'dish_empty': (255, 255, 0),       # Cyan (BGR)
        'dish_kakigori': (255, 79, 255),    # Magenta (BGR)  
        'dish_not_empty': (0, 255, 0),     # Green (BGR)
        'tray_empty': (0, 255, 255),       # Yellow (BGR)
        'tray_kakigori': (0, 165, 255),    # Orange (BGR)
        'tray_not_empty': (30, 30, 220)      # Red (BGR)
    }
    
    # Video Processing Settings
    MAX_WIDTH = 1300
    JPEG_QUALITY = 100
    PROCESSING_TIMES_BUFFER = 30
    
    # Bounding Box Settings - Easy to modify for future customization
    BBOX_THICKNESS = 2              # Thickness of bounding box lines
    BBOX_FONT_SCALE = 0.7          # Font size for labels (0.5 = small, 0.7 = medium, 1.0 = large)
    BBOX_FONT_THICKNESS = 2        # Font thickness for labels (1 = thin, 2 = medium, 3 = thick)
    BBOX_TEXT_PADDING = 10         # Padding around text background
    BBOX_TEXT_OFFSET_X = 2         # Horizontal text offset
    BBOX_TEXT_OFFSET_Y = 6         # Vertical text offset from top of box
    
    # Threading Settings
    BUFFER_SIZE = 1
    MAX_RETRIES = 3
    
    # Database Settings
    FEEDBACK_DB_PATH = "../feedback.db"
    
    # Flask Settings
    HOST = '0.0.0.0'
    PORT = 5002
    DEBUG = False
    THREADED = True
    
    # Video Encoding Settings
    FRAME_DELAY = 0.01
    RETRY_DELAY = 0.5 