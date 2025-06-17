#!/usr/bin/env python3
"""
Flask Routes Module
Contains all API endpoints and web routes for the video streaming application
"""

from flask import Flask, Response, render_template, jsonify, request
import json
import time
from datetime import datetime

# Global variables for Flask integration
video_processor = None
feedback_manager = None


def configure_routes(app, video_proc, feedback_mgr):
    """Configure all routes for the Flask application"""
    global video_processor, feedback_manager
    video_processor = video_proc
    feedback_manager = feedback_mgr
    
    # Disable Flask request logging
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('index.html')
    
    @app.route('/feedback')
    def feedback():
        """Feedback management page"""
        feedback_list = feedback_manager.get_all_feedback(limit=50)
        return render_template('feedback.html', feedback_list=feedback_list)
    
    @app.route('/video_feed')
    def video_feed():
        """Video streaming endpoint"""
        def generate():
            for frame in video_processor.video_stream_generator():
                yield frame
        
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/stats')
    def stats():
        """Get performance statistics"""
        return jsonify(video_processor.get_stats())
    
    @app.route('/health')
    def health_check():
        """Simple health check endpoint for Docker"""
        try:
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'service': 'dispatch-monitoring'
            })
        except Exception:
            return jsonify({
                'status': 'unhealthy'
            }), 500
    
    # API Routes
    @app.route('/api/toggle_playback', methods=['POST'])
    def api_toggle_playback():
        """Toggle video playback (play/pause)"""
        try:
            is_playing = video_processor.toggle_playback()
            return jsonify({
                'success': True,
                'is_playing': is_playing,
                'message': 'Playing' if is_playing else 'Paused'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/seek_to_frame', methods=['POST'])
    def api_seek_to_frame():
        """Seek to a specific frame"""
        try:
            data = request.get_json()
            frame_number = data.get('frame_number', 0)
            
            success = video_processor.seek_to_frame(frame_number)
            
            return jsonify({
                'success': success,
                'frame_number': frame_number,
                'message': f'Seeked to frame {frame_number}' if success else 'Invalid frame number'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/view_all_feedback')
    def api_view_all_feedback():
        """Get all feedback data"""
        try:
            feedback_type = request.args.get('type')
            limit = int(request.args.get('limit', 100))
            offset = int(request.args.get('offset', 0))
            
            feedback_list = feedback_manager.get_all_feedback(
                limit=limit, 
                offset=offset, 
                feedback_type=feedback_type
            )
            
            feedback_stats = feedback_manager.get_feedback_stats()
            
            return jsonify({
                'success': True,
                'feedbacks': feedback_list,
                'count': len(feedback_list),
                'stats': feedback_stats
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/submit_feedback', methods=['POST'])
    def api_submit_feedback():
        """Submit user feedback"""
        try:
            data = request.get_json()
            
            # Get current video state
            stats = video_processor.get_stats()
            current_frame = stats['video_info']['current_frame']
            
            # Extract feedback data
            feedback_text = data.get('feedback_text', '').strip()
            feedback_type = data.get('feedback_type', 'general')
            rating = data.get('rating')
            user_corrections = data.get('user_corrections', '').strip()
            
            # Validate required fields
            if not feedback_text:
                return jsonify({
                    'success': False,
                    'error': 'Feedback text is required'
                }), 400
            
            # Add feedback to database
            feedback_id = feedback_manager.add_feedback(
                frame_number=current_frame,
                feedback_type=feedback_type,
                feedback_text=feedback_text,
                rating=rating,
                user_corrections=user_corrections if user_corrections else None,
                video_file=video_processor.video_path,
                processing_time=stats.get('avg_processing_time_ms'),
                detection_count=stats.get('current_detections'),
                class_counts=stats.get('class_counts'),
                user_ip=request.remote_addr,
                session_id=request.headers.get('User-Agent', '')[:100]
            )
            
            if feedback_id:
                return jsonify({
                    'success': True,
                    'feedback_id': feedback_id,
                    'message': 'Feedback submitted successfully!',
                    'frame_number': current_frame
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to save feedback'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/feedback/delete/<int:feedback_id>', methods=['DELETE'])
    def api_delete_feedback(feedback_id):
        """Delete specific feedback"""
        try:
            success = feedback_manager.delete_feedback(feedback_id)
            
            return jsonify({
                'success': success,
                'message': f'Feedback {feedback_id} deleted' if success else 'Feedback not found'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/feedback/export')
    def api_export_feedback():
        """Export feedback to CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feedback_export_{timestamp}.csv"
            
            success = feedback_manager.export_feedback_csv(filename)
            
            return jsonify({
                'success': success,
                'filename': filename if success else None,
                'message': f'Feedback exported to {filename}' if success else 'Export failed'
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/system/status')
    def api_system_status():
        """Get system health status"""
        try:
            stats = video_processor.get_stats()
            feedback_stats = feedback_manager.get_feedback_stats()
            
            return jsonify({
                'success': True,
                'system_status': {
                    'video_processor': {
                        'running': video_processor.running,
                        'models_loaded': stats['models_loaded'],
                        'fps': stats['fps'],
                        'current_frame': stats['video_info']['current_frame'],
                        'total_frames': stats['video_info']['total_frames']
                    },
                    'feedback_system': {
                        'total_feedback': feedback_stats['total_feedback'],
                        'recent_feedback': feedback_stats['recent_feedback_24h'],
                        'average_rating': feedback_stats['average_rating']
                    },
                    'timestamp': datetime.now().isoformat()
                }
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/video/info')
    def api_video_info():
        """Get detailed video information"""
        try:
            stats = video_processor.get_stats()
            
            return jsonify({
                'success': True,
                'video_info': {
                    'path': video_processor.video_path,
                    'total_frames': stats['video_info']['total_frames'],
                    'current_frame': stats['video_info']['current_frame'],
                    'fps': stats['video_info']['video_fps'],
                    'is_playing': stats['video_info']['is_playing'],
                    'duration_seconds': stats['video_info']['total_frames'] / stats['video_info']['video_fps'],
                    'progress_percentage': (stats['video_info']['current_frame'] / stats['video_info']['total_frames']) * 100
                },
                'processing_info': {
                    'detection_model': video_processor.detection_model_path,
                    'classification_model': video_processor.classification_model_path,
                    'inference_area': video_processor.inference_area,
                    'confidence_threshold': video_processor.confidence_threshold
                }
            })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/video/list')
    def api_list_videos():
        """Get list of available video files"""
        try:
            import os
            from pathlib import Path
            
            # Common video extensions
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
            
            # Search in current directory and common video directories
            search_paths = [
                Path('.'),
                Path('..'),
                Path('../videos'),
                Path('videos'),
                Path('../data'),
                Path('data')
            ]
            
            videos = []
            for search_path in search_paths:
                if search_path.exists():
                    for file_path in search_path.rglob('*'):
                        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
                            # Get relative path for display
                            try:
                                rel_path = str(file_path.relative_to(Path('.')))
                            except ValueError:
                                rel_path = str(file_path)
                            
                            videos.append({
                                'name': file_path.name,
                                'path': rel_path,
                                'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
                                'is_current': rel_path == video_processor.video_path
                            })
            
            # Remove duplicates and sort
            unique_videos = {v['path']: v for v in videos}.values()
            sorted_videos = sorted(unique_videos, key=lambda x: x['name'])
            
            return jsonify({
                'success': True,
                'videos': sorted_videos,
                'current_video': video_processor.video_path
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    @app.route('/api/video/change', methods=['POST'])
    def api_change_video():
        """Change the current video"""
        try:
            data = request.get_json()
            new_video_path = data.get('video_path')
            
            if not new_video_path:
                return jsonify({
                    'success': False,
                    'error': 'Video path is required'
                }), 400
            
            # Check if video file exists
            from pathlib import Path
            if not Path(new_video_path).exists():
                return jsonify({
                    'success': False,
                    'error': f'Video file not found: {new_video_path}'
                }), 404
            
            # Pause the current video
            video_processor.is_playing = False
            
            # Clear video cache and reset
            with video_processor.cap_lock:
                if video_processor.cap:
                    video_processor.cap.release()
                    video_processor.cap = None
            
            # Update video path
            video_processor.video_path = new_video_path
            
            # Reload video
            if video_processor.load_video():
                # Reset frame position
                video_processor.current_frame_number = 0
                video_processor.target_frame = 0
                
                # Clear processing times cache for system stability
                video_processor.processing_times = []
                video_processor.frame_count = 0
                video_processor.fps_counter = 0
                video_processor.fps_time = time.time()
                
                return jsonify({
                    'success': True,
                    'message': f'Video changed to: {new_video_path}',
                    'video_info': {
                        'path': video_processor.video_path,
                        'total_frames': video_processor.total_frames,
                        'video_fps': video_processor.video_fps
                    }
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'Failed to load the new video'
                }), 500
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500
    
    return app 