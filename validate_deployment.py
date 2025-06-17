#!/usr/bin/env python3
"""
Deployment Validation Script
Validates that all components are properly deployed and functional
"""

import os
import sys
import time
import requests
import subprocess
import json
from pathlib import Path

def print_status(message, status="INFO"):
    """Print formatted status message"""
    symbols = {
        "INFO": "ℹ️",
        "SUCCESS": "✅",
        "ERROR": "❌",
        "WARNING": "⚠️"
    }
    print(f"{symbols.get(status, 'ℹ️')} {message}")

def check_docker_running():
    """Check if Docker is running"""
    try:
        result = subprocess.run(['docker', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_status("Docker is running", "SUCCESS")
            return True
        else:
            print_status("Docker is not running", "ERROR")
            return False
    except FileNotFoundError:
        print_status("Docker command not found", "ERROR")
        return False

def check_docker_compose():
    """Check if docker-compose is available"""
    try:
        result = subprocess.run(['docker-compose', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print_status("Docker Compose is available", "SUCCESS")
            return True
        else:
            print_status("Docker Compose is not available", "ERROR")
            return False
    except FileNotFoundError:
        print_status("Docker Compose command not found", "ERROR")
        return False

def check_required_files():
    """Check if all required files exist"""
    required_files = [
        'Dockerfile',
        'docker-compose.yml',
        'requirements.txt',
        'run_video_streaming_app.py',
        'video_streaming_app/core/config.py',
        'video_streaming_app/core/video_processor.py',
        'video_streaming_app/routes/api_routes.py'
    ]
    
    all_files_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print_status(f"Found: {file_path}", "SUCCESS")
        else:
            print_status(f"Missing: {file_path}", "ERROR")
            all_files_exist = False
    
    return all_files_exist

def check_container_status():
    """Check if containers are running"""
    try:
        result = subprocess.run(['docker-compose', 'ps'], capture_output=True, text=True)
        if result.returncode == 0:
            print_status("Container status check completed", "SUCCESS")
            print(result.stdout)
            return True
        else:
            print_status("Failed to check container status", "ERROR")
            return False
    except Exception as e:
        print_status(f"Error checking containers: {e}", "ERROR")
        return False

def check_web_interface():
    """Check if web interface is accessible"""
    urls_to_check = [
        ('http://localhost:5002', 'Main Application'),
        ('http://localhost:5002/health', 'Health Check'),
        ('http://localhost:5002/api/system/status', 'System Status API')
    ]
    
    all_accessible = True
    for url, description in urls_to_check:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                print_status(f"{description} is accessible", "SUCCESS")
            else:
                print_status(f"{description} returned status {response.status_code}", "WARNING")
                all_accessible = False
        except requests.exceptions.ConnectionError:
            print_status(f"{description} is not accessible (connection refused)", "ERROR")
            all_accessible = False
        except requests.exceptions.Timeout:
            print_status(f"{description} timed out", "ERROR")
            all_accessible = False
        except Exception as e:
            print_status(f"{description} check failed: {e}", "ERROR")
            all_accessible = False
    
    return all_accessible

def check_gpu_support():
    """Check if GPU support is available in container"""
    try:
        result = subprocess.run([
            'docker-compose', 'exec', '-T', 'dispatch-monitoring', 
            'python', '-c', 'import torch; print(f"CUDA available: {torch.cuda.is_available()}")'
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and 'CUDA available: True' in result.stdout:
            print_status("GPU support is available", "SUCCESS")
            return True
        else:
            print_status("GPU support not available (CPU mode will be used)", "WARNING")
            return False
    except Exception as e:
        print_status(f"Could not check GPU support: {e}", "WARNING")
        return False

def main():
    """Main validation routine"""
    print("=" * 60)
    print("🐳 DISPATCH MONITORING SYSTEM - DEPLOYMENT VALIDATION")
    print("=" * 60)
    
    # Check prerequisites
    print("\n📋 Checking Prerequisites...")
    docker_ok = check_docker_running()
    compose_ok = check_docker_compose()
    
    if not (docker_ok and compose_ok):
        print_status("Prerequisites not met. Please install Docker and Docker Compose.", "ERROR")
        sys.exit(1)
    
    # Check required files
    print("\n📁 Checking Required Files...")
    files_ok = check_required_files()
    
    if not files_ok:
        print_status("Some required files are missing. Please check the file structure.", "ERROR")
        sys.exit(1)
    
    # Check container status
    print("\n🐳 Checking Container Status...")
    check_container_status()
    
    # Check web interface
    print("\n🌐 Checking Web Interface...")
    web_ok = check_web_interface()
    
    # Check GPU support
    print("\n🔧 Checking GPU Support...")
    check_gpu_support()
    
    # Final summary
    print("\n" + "=" * 60)
    if web_ok:
        print_status("✨ DEPLOYMENT VALIDATION COMPLETED SUCCESSFULLY! ✨", "SUCCESS")
        print("\n🎯 Access Points:")
        print("  • Main Application: http://localhost:5002")
        print("  • Health Check: http://localhost:5002/health")
        print("  • System Status: http://localhost:5002/api/system/status")
        print("  • Feedback System: http://localhost:5002/feedback")
    else:
        print_status("Deployment validation completed with warnings", "WARNING")
        print("Please check the issues above and ensure all services are running properly.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 