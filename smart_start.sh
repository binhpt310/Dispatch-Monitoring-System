#!/bin/bash

# Smart Startup Script for New Dispatch Monitoring System
# Automatically detects GPU support and starts with optimal configuration

echo "-  New Dispatch Monitoring System - Smart Startup"
echo "=================================================="

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "-  Docker is not running. Please start Docker first."
        exit 1
    fi
    echo "-  Docker is running"
}

# Function to check if NVIDIA Docker runtime is available
check_gpu_support() {
    echo "-  Checking GPU support..."
    
    # Check if nvidia-docker runtime is available
    if docker info 2>/dev/null | grep -q nvidia; then
        echo "-  NVIDIA Docker runtime detected"
        
        # Check if nvidia-smi is available on host
        if command -v nvidia-smi > /dev/null 2>&1; then
            echo "-  NVIDIA drivers detected on host:"
            nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "    Could not query GPU details"
        else
            echo "-  NVIDIA drivers not found in PATH (this is common on Windows)"
        fi
        
        # Test if we can actually run a GPU container with more detailed error reporting
        echo "-  Testing GPU container access..."
        if docker run --rm --gpus all nvidia/cuda:11.8.0-devel-ubuntu22.04 nvidia-smi > /tmp/gpu_test.log 2>&1; then
            echo "-  GPU access confirmed"
            if [ -f /tmp/gpu_test.log ]; then
                echo "-  GPU Details:"
                cat /tmp/gpu_test.log | head -3 | sed 's/^/    /'
                rm -f /tmp/gpu_test.log
            fi
            return 0
        else
            echo "-  GPU runtime available but GPU access failed"
            echo "-  Error details:"
            if [ -f /tmp/gpu_test.log ]; then
                cat /tmp/gpu_test.log | sed 's/^/    /' | head -10
                rm -f /tmp/gpu_test.log
            fi
            echo ""
            echo "-  Common solutions:"
            echo "    1. Ensure NVIDIA drivers are installed on host"
            echo "    2. Install NVIDIA Container Toolkit"
            echo "    3. On Windows: Enable GPU support in Docker Desktop"
            echo "    4. Restart Docker service after installing GPU support"
            return 1
        fi
    else
        echo "-  NVIDIA Docker runtime not found"
        echo "-  To enable GPU support:"
        echo "    1. Install NVIDIA Container Toolkit"
        echo "    2. Configure Docker to use nvidia runtime"
        echo "    3. Restart Docker service"
        return 1
    fi
}

# Function to display GPU status
display_gpu_status() {
    if check_gpu_support; then
        echo "-  GPU support detected - Application will use GPU acceleration"
        echo "-  Your RTX 4060 will accelerate AI model inference"
    else
        echo "-  No GPU support - Application will fall back to CPU mode"
        echo "-  Performance will be slower but functional"
    fi
}

# Function to clean up any previous containers
cleanup_containers() {
    echo "-  Cleaning up previous containers..."
    docker-compose down > /dev/null 2>&1
    echo "-  Cleanup completed"
}

# Function to build and start the application
start_application() {
    echo "-  Building application..."
    docker-compose build
    
    echo "-  Starting dispatch monitoring application..."
    docker-compose up -d dispatch-monitoring
    
    # Wait for application to start
    echo "-  Waiting for application to start..."
    sleep 15
    
    # Check if application is running
    if docker-compose ps | grep -q "dispatch-monitoring.*Up"; then
        echo "-  Application started successfully!"
        
        # Display device information from logs
        echo ""
        echo "-  Device Detection Result:"
        docker-compose logs dispatch-monitoring | grep -E "(Using device|CUDA|CPU)" | tail -5
        
        echo ""
        echo "-  Access the application at:"
        echo "  Main App: http://localhost:5002"
        echo "  Health Check: http://localhost:5002/health"
        echo ""
        echo "-  Management commands:"
        echo "  View logs: docker-compose logs -f dispatch-monitoring"
        echo "  Stop app: docker-compose down"
        echo ""
        echo "-  Training commands:"
        echo "  Detection: docker-compose --profile training up training-detection"
        echo "  Classification: docker-compose --profile training up training-classification"
        
    else
        echo "-  Failed to start application. Checking logs..."
        docker-compose logs dispatch-monitoring
        return 1
    fi
}

# Main execution flow
main() {
    check_docker
    cleanup_containers
    display_gpu_status
    start_application
}

# Run main function
main "$@" 