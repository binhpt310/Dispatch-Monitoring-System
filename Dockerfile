# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-dev \
    libcairo-gobject2 \
    libgdk-pixbuf2.0-dev \
    libpango1.0-dev \
    libatk1.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional system packages for OpenCV
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire application
COPY . .

# Create necessary directories
RUN mkdir -p models results logs feedback

# Download base YOLO models if not present
RUN python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt'); YOLO('yolo11m-cls.pt')" || true

# Set proper permissions
RUN chmod +x run_video_streaming_app.py
RUN chmod -R 755 /app

# Expose the port the app runs on
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5002/health || exit 1

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting New Dispatch Monitoring System..."\n\
echo "System Information:"\n\
echo "  - Python: $(python --version)"\n\
echo "  - CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits || echo \"Not available\")"\n\
echo "  - GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits || echo \"Not available\")"\n\
echo "  - Working Directory: $(pwd)"\n\
echo "  - Available Models: $(ls -la models/ || echo \"No models directory\")"\n\
echo ""\n\
echo "Checking dependencies..."\n\
python -c "import ultralytics, cv2, torch, flask; print(\"All dependencies loaded successfully\")" || exit 1\n\
echo ""\n\
echo "Starting Video Streaming Application on port 5002..."\n\
exec python run_video_streaming_app.py\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"] 