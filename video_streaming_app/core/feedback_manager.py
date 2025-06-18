#!/usr/bin/env python3
"""
Feedback Management Module
Handles database operations for user feedback and corrections
"""

import sqlite3
import time
from datetime import datetime
from pathlib import Path

from .config import Config


class FeedbackManager:
    """Database manager for user feedback and corrections"""
    
    def __init__(self, db_path=None):
        """Initialize feedback manager with database path"""
        self.db_path = db_path or Config.FEEDBACK_DB_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize the feedback database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create enhanced feedback table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        frame_number INTEGER,
                        feedback_type TEXT,
                        feedback_text TEXT,
                        rating INTEGER,
                        user_corrections TEXT,
                        video_file TEXT,
                        processing_time REAL,
                        detection_count INTEGER,
                        class_counts TEXT,
                        user_ip TEXT,
                        session_id TEXT
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_feedback_timestamp 
                    ON feedback(timestamp)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_feedback_frame 
                    ON feedback(frame_number)
                ''')
                
                cursor.execute('''
                    CREATE INDEX IF NOT EXISTS idx_feedback_type 
                    ON feedback(feedback_type)
                ''')
                
                conn.commit()
                print(f"Feedback database initialized: {self.db_path}")
                
        except Exception as e:
            print(f"Error initializing feedback database: {e}")
    
    def add_feedback(self, frame_number, feedback_type, feedback_text, 
                    rating=None, user_corrections=None, video_file=None,
                    processing_time=None, detection_count=None, class_counts=None,
                    user_ip=None, session_id=None):
        """Add feedback to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Convert class_counts to string if it's a dict
                class_counts_str = str(class_counts) if class_counts else None
                
                cursor.execute('''
                    INSERT INTO feedback (
                        frame_number, feedback_type, feedback_text, rating,
                        user_corrections, video_file, processing_time,
                        detection_count, class_counts, user_ip, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    frame_number, feedback_type, feedback_text, rating,
                    user_corrections, video_file, processing_time,
                    detection_count, class_counts_str, user_ip, session_id
                ))
                
                conn.commit()
                feedback_id = cursor.lastrowid
                
                print(f"Feedback added successfully (ID: {feedback_id})")
                return feedback_id
                
        except Exception as e:
            print(f"Error adding feedback: {e}")
            return None
    
    def get_all_feedback(self, limit=100, offset=0, feedback_type=None):
        """Get all feedback from the database with optional filtering"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query with optional type filter
                query = '''
                    SELECT id, timestamp, frame_number, feedback_type, 
                           feedback_text, rating, user_corrections, video_file,
                           processing_time, detection_count, class_counts
                    FROM feedback
                '''
                params = []
                
                if feedback_type:
                    query += ' WHERE feedback_type = ?'
                    params.append(feedback_type)
                
                query += ' ORDER BY timestamp DESC LIMIT ? OFFSET ?'
                params.extend([limit, offset])
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                feedback_list = []
                for row in rows:
                    feedback_list.append({
                        'id': row[0],
                        'timestamp': row[1],
                        'frame_number': row[2],
                        'feedback_type': row[3],
                        'feedback_text': row[4],
                        'rating': row[5],
                        'user_corrections': row[6],
                        'video_file': row[7],
                        'processing_time': row[8],
                        'detection_count': row[9],
                        'class_counts': row[10]
                    })
                
                return feedback_list
                
        except Exception as e:
            print(f"Error retrieving feedback: {e}")
            return []
    
    def get_feedback_stats(self):
        """Get feedback statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get total feedback count
                cursor.execute('SELECT COUNT(*) FROM feedback')
                total_count = cursor.fetchone()[0]
                
                # Get feedback by type
                cursor.execute('''
                    SELECT feedback_type, COUNT(*) 
                    FROM feedback 
                    GROUP BY feedback_type
                ''')
                feedback_by_type = dict(cursor.fetchall())
                
                # Get average rating
                cursor.execute('SELECT AVG(rating) FROM feedback WHERE rating IS NOT NULL')
                avg_rating = cursor.fetchone()[0]
                
                # Get recent feedback count (last 24 hours)
                cursor.execute('''
                    SELECT COUNT(*) FROM feedback 
                    WHERE timestamp >= datetime('now', '-1 day')
                ''')
                recent_count = cursor.fetchone()[0]
                
                return {
                    'total_feedback': total_count,
                    'feedback_by_type': feedback_by_type,
                    'average_rating': round(avg_rating, 2) if avg_rating else None,
                    'recent_feedback_24h': recent_count
                }
                
        except Exception as e:
            print(f"Error getting feedback stats: {e}")
            return {
                'total_feedback': 0,
                'feedback_by_type': {},
                'average_rating': None,
                'recent_feedback_24h': 0
            }
    
    def delete_feedback(self, feedback_id):
        """Delete specific feedback by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM feedback WHERE id = ?', (feedback_id,))
                conn.commit()
                
                if cursor.rowcount > 0:
                    print(f"Feedback {feedback_id} deleted successfully")
                    return True
                else:
                    print(f"Feedback {feedback_id} not found")
                    return False
                    
        except Exception as e:
            print(f"Error deleting feedback: {e}")
            return False
    
    def cleanup_old_feedback(self, days_old=30):
        """Clean up feedback older than specified days"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM feedback 
                    WHERE timestamp < datetime('now', '-{} days')
                '''.format(days_old))
                conn.commit()
                
                deleted_count = cursor.rowcount
                print(f"Cleaned up {deleted_count} old feedback entries")
                return deleted_count
                
        except Exception as e:
            print(f"Error cleaning up feedback: {e}")
            return 0
    
    def export_feedback_csv(self, output_file="feedback_export.csv"):
        """Export all feedback to CSV file"""
        try:
            import csv
            
            feedback_data = self.get_all_feedback(limit=10000)  # Get all feedback
            
            if not feedback_data:
                print("No feedback data to export")
                return False
            
            with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'id', 'timestamp', 'frame_number', 'feedback_type',
                    'feedback_text', 'rating', 'user_corrections', 'video_file',
                    'processing_time', 'detection_count', 'class_counts'
                ]
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for feedback in feedback_data:
                    writer.writerow(feedback)
            
            print(f"Feedback exported to {output_file}")
            return True
            
        except Exception as e:
            print(f"Error exporting feedback: {e}")
            return False 