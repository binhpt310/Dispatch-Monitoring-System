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
        
        # Test if we can actually run a GPU container
        if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
            echo "-  GPU access confirmed"
            return 0
        else
            echo "-   GPU runtime available but GPU access failed"
            return 1
        fi
    else
        echo "-   NVIDIA Docker runtime not found"
        return 1
    fi
}

# Function to start with GPU support
start_with_gpu() {
    echo "-  Starting with GPU support..."
    
    # Create temporary docker-compose override for GPU
    cat > docker-compose.override.yml << EOF
version: '3.8'
services:
  dispatch-monitoring:
    runtime: nvidia
    environment:
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  training-detection:
    runtime: nvidia
    environment:
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  training-classification:
    runtime: nvidia
    environment:
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
EOF

    echo "-  GPU configuration applied"
}

# Function to start with CPU only
start_with_cpu() {
    echo "-  Starting with CPU-only mode..."
    
    # Remove any existing override file to ensure CPU mode
    rm -f docker-compose.override.yml
    
    echo "-  CPU-only configuration applied"
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
    
    if check_gpu_support; then
        echo "-  GPU support detected - Using GPU acceleration"
        start_with_gpu
    else
        echo "-  No GPU support - Using CPU mode (slower but functional)"
        start_with_cpu
    fi
    
    start_application
    
    # Cleanup the override file after successful start
    if [ -f docker-compose.override.yml ]; then
        echo ""
        echo "-   GPU configuration file created. To remove GPU support, run:"
        echo "   rm docker-compose.override.yml && docker-compose down && docker-compose up -d"
    fi
}

# Run main function
main "$@" 