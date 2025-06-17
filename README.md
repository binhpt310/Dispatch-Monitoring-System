# 🍽️ Dispatch Monitoring System

A computer vision system for monitoring dish and tray status in restaurant/food service environments using YOLO11 detection and classification models.

## 📋 System Overview

This system provides real-time monitoring of dishes and trays with two main components:
- **Desktop Application**: Full-featured Tkinter GUI with advanced controls
- **Web Application**: Browser-based interface with real-time processing

### 🎯 Detection Classes
- **Detection Model**: `dish`, `tray` (2 classes)
- **Classification Model**: `dish_empty`, `dish_kakigori`, `dish_not_empty`, `tray_empty`, `tray_not_empty`, `tray_kakigori` (6 classes)

## 🚀 Quick Start with Docker Compose

### Prerequisites
- **Docker**: 20.10+ with GPU support
- **Docker Compose**: 2.0+
- **NVIDIA GPU**: With CUDA support (recommended)
- **System**: 8GB+ RAM, 10GB+ storage

### 1. Docker Setup
```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Container Toolkit (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Test GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi
```

### 2. Clone and Deploy
```bash
# Clone the repository
git clone <repository-url>
cd New_Dispatch_Monitoring_System

# Deploy the complete system
docker-compose up -d
```

### 3. Choose Your Workflow

#### Option A: Use Pre-trained Models (Quick Start)
```bash
# Web Application (recommended)
docker-compose exec webapp python app.py
# Access at: http://localhost:5000

# Desktop Application
docker-compose exec desktop python desktop_app.py
# Requires X11 forwarding for GUI
```

#### Option B: Train Models from Scratch
```bash
# Train detection model
docker-compose exec training python training/train_detection.py

# Train classification model  
docker-compose exec training python training/train_classification.py

# High-resolution training
docker-compose exec training python training/train_highres.py

# Monitor training progress
docker-compose logs -f training
```

## 🐳 Docker Services

### Service Architecture
```yaml
services:
  webapp:      # Web application with Flask
  desktop:     # Desktop application with Tkinter
  training:    # Model training environment
  nginx:       # Reverse proxy (production)
```

### Container Features
- **GPU Acceleration**: CUDA support in all containers
- **Volume Mounting**: Persistent data storage
- **Hot Reload**: Development mode with live updates
- **Resource Limits**: Optimized memory and GPU usage

## 🔧 Container-Based Training

### Training in Containers
```bash
# Start training container
docker-compose up training

# Interactive training session
docker-compose exec training bash

# Custom training parameters
docker-compose exec training python training/train_detection.py --epochs 50 --batch-size 8

# Monitor GPU usage
docker-compose exec training nvidia-smi
```

### Training Data Setup
```bash
# Mount your dataset
# Edit docker-compose.yml to point to your dataset:
volumes:
  - ./your_dataset:/app/Dataset
  - ./models:/app/models
  - ./results:/app/results
```

## 🌐 Web Application (Container)

### Deployment
```bash
# Start web service
docker-compose up webapp

# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# Scale web service
docker-compose up --scale webapp=3
```

### Web Features in Container
- **Background Model Loading**: Non-blocking startup
- **Smart Caching**: 50-frame cache system
- **Real-time Processing**: < 1 second video loading
- **GPU Acceleration**: Automatic CUDA detection
- **Volume Persistence**: Uploaded videos and results saved

### Access Points
- **Web Interface**: http://localhost:5000
- **API Endpoints**: http://localhost:5000/api/*
- **Health Check**: http://localhost:5000/health

## 🖥️ Desktop Application (Container)

### X11 Forwarding Setup
```bash
# Linux - Enable X11 forwarding
xhost +local:docker

# Run desktop app with GUI
docker-compose run --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix desktop python desktop_app.py

# Alternative: VNC access
docker-compose up desktop-vnc
# Access via VNC viewer at localhost:5901
```

### Desktop Features in Container
- **Full GUI**: Complete Tkinter interface
- **GPU Processing**: CUDA acceleration
- **File Access**: Mounted volumes for videos/results
- **Real-time Inference**: Optimized performance

## 📊 Container Monitoring

### Health Checks
```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs webapp
docker-compose logs training
docker-compose logs desktop

# Resource usage
docker stats
```

### Performance Monitoring
```bash
# GPU usage across containers
docker-compose exec webapp nvidia-smi
docker-compose exec training nvidia-smi

# Container resource limits
docker-compose exec webapp cat /sys/fs/cgroup/memory/memory.limit_in_bytes
```

## 🔧 Development Mode

### Local Development with Containers
```bash
# Development mode with hot reload
docker-compose -f docker-compose.dev.yml up

# Mount local code for development
volumes:
  - ./webapp:/app/webapp
  - ./training:/app/training
  - ./models:/app/models
```

### Debugging
```bash
# Interactive debugging
docker-compose exec webapp bash
docker-compose exec training python -m pdb training/train_detection.py

# View container logs in real-time
docker-compose logs -f --tail=100 webapp
```

## 📁 Container Volume Structure

```
New_Dispatch_Monitoring_System/
├── docker-compose.yml              # Main orchestration
├── docker-compose.prod.yml         # Production config
├── docker-compose.dev.yml          # Development config
├── Dockerfile.webapp               # Web app container
├── Dockerfile.desktop              # Desktop app container
├── Dockerfile.training             # Training container
├── models/                         # Mounted: Trained models
├── Dataset/                        # Mounted: Training data
├── results/                        # Mounted: Training outputs
├── webapp/uploads/                 # Mounted: Uploaded videos
└── logs/                          # Mounted: Container logs
```

## 🚀 Production Deployment

### Production Configuration
```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# With load balancer
docker-compose -f docker-compose.prod.yml --scale webapp=3 up -d

# SSL/HTTPS setup
# Edit docker-compose.prod.yml for SSL certificates
```

### Environment Variables
```bash
# Create .env file
cat > .env << EOF
CUDA_VISIBLE_DEVICES=0
FLASK_ENV=production
MODEL_PATH=/app/models
UPLOAD_PATH=/app/uploads
MAX_WORKERS=4
EOF
```

## 🔍 Troubleshooting Containers

### Common Issues
```bash
# GPU not available in container
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Permission issues
sudo chown -R $USER:$USER ./models ./results ./Dataset

# Container memory issues
docker-compose down
docker system prune -f
docker-compose up -d

# Port conflicts
docker-compose down
sudo netstat -tulpn | grep :5000
docker-compose up -d
```

### Container Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs webapp
docker-compose logs training

# Follow logs
docker-compose logs -f webapp
```

## 📈 Performance Optimization

### Container Resource Limits
```yaml
# In docker-compose.yml
services:
  webapp:
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Scaling
```bash
# Horizontal scaling
docker-compose up --scale webapp=3

# Load balancing with nginx
docker-compose -f docker-compose.prod.yml up -d
```

## 🛠️ Manual Installation (Alternative)

<details>
<summary>Click to expand manual installation (not recommended)</summary>

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 2GB+ storage space

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements_desktop.txt
```

### Running Applications
```bash
# Desktop application
python desktop_app.py

# Web application
python run_webapp.py
```

</details>

## 🎮 Usage Guide

### Container-Based Usage
- **Web Interface**: Upload videos, adjust settings, view results
- **Desktop GUI**: Full-featured interface with advanced controls
- **Training**: Custom model training with your datasets
- **API Access**: RESTful endpoints for integration

### Key Features
- **Real-time Processing**: < 1 second video loading
- **GPU Acceleration**: Automatic CUDA detection
- **Scalable**: Multi-container deployment
- **Persistent**: Data saved across container restarts

## 🤝 Contributing

### Development Workflow
```bash
# Fork and clone
git clone <your-fork>
cd New_Dispatch_Monitoring_System

# Development environment
docker-compose -f docker-compose.dev.yml up -d

# Make changes and test
docker-compose exec webapp python -m pytest

# Submit pull request
```

## 📄 License

[Add your license information here]

## 🆘 Support

For issues and questions:
1. Check container logs: `docker-compose logs`
2. Verify GPU support: `docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi`
3. Review troubleshooting section
4. Contact development team

---

**Deployment Status**: 🐳 **Docker-First Architecture** | ✅ **Container Ready** | 🚀 **Production Scalable** 