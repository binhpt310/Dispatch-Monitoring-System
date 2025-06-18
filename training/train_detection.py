import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.training_config import DetectionConfig
from utils.data_utils import DatasetValidator


class DetectionTrainer:

    
    def __init__(self):
        self.model = None
        self.training_results = None
        
        # Device detection and configuration
        self.device = self._detect_device()
        print(f"- Training device: {self.device}")
        
        # Configuration with CPU/GPU optimization
        self.epochs = 100
        self.image_size = 640  # Standard detection image size
        self.learning_rate = 0.01
        self.workers = 8
        self.patience = 50
        
        # Optimize for device
        if self.device == 'cpu':
            # CPU optimized settings
            self.batch_size = 4  # Smaller batch for CPU
            self.image_size = 640  # Standard detection image size
            self.workers = min(4, os.cpu_count())  # Limit workers on CPU
            print("- CPU mode: Using reduced batch size for better performance")
        else:
            # GPU optimized settings
            self.batch_size = 16
            self.image_size = 640  # Standard detection image size
        
        # Paths
        self.dataset_yaml = Path("Dataset/detection/dataset.yaml")
        self.results_dir = Path("results")
        self.models_dir = Path("models")
        self.base_model = "yolo12s.pt"
        
        self.validate_environment()
    
    def _detect_device(self):
        """Detect available device (CUDA/CPU) and return appropriate device string"""
        try:
            # Check for forced CPU mode via environment variable
            if os.getenv('FORCE_CPU', '').lower() in ['true', '1', 'yes']:
                cpu_count = os.cpu_count()
                print(f"-  FORCE_CPU enabled - Using CPU mode ({cpu_count} cores)")
                return 'cpu'
            
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"-  CUDA available: {gpu_count} GPU(s) - Using {gpu_name} ({total_memory:.1f}GB)")
                return 'cuda'
            else:
                cpu_count = os.cpu_count()
                print(f"-   CUDA not available - Using CPU ({cpu_count} cores)")
                print("-  CPU training will be slower but still functional")
                return 'cpu'
        except Exception as e:
            print(f"-   Device detection error: {e} - Falling back to CPU")
            return 'cpu'
    
    def generate_model_name(self) -> str:
        # Extract model name without extension
        base_name = Path(self.base_model).stem
        
        # Get current date in DDMMYYYY format
        current_date = datetime.now().strftime("%d%m%Y")
        
        # Add device suffix for clarity
        device_suffix = "gpu" if self.device == 'cuda' else "cpu"
        
        # Create dynamic name: detection_model_{base_name}_best_{epochs}_{device}_{date}
        model_name = f"detection_model_{base_name}_best_{self.epochs}_{device_suffix}_{current_date}"
        
        return model_name
    

    def validate_environment(self):
        print("-  Validating training environment...")
        
        # Print device information
        if self.device == 'cuda':
            gpu_count = torch.cuda.device_count()
            print(f"-  CUDA available with {gpu_count} GPU(s)")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            cpu_count = os.cpu_count()
            print(f"-  Using CPU with {cpu_count} cores")
            print("⏰ Note: CPU training will take significantly longer than GPU training")
        
        # Check base model
        if not self.dataset_yaml.exists():
            print(f"-  Dataset YAML not found: {self.dataset_yaml}")
            raise FileNotFoundError(f"Dataset YAML not found: {self.dataset_yaml}")
        
        # Validate dataset
        validator = DatasetValidator(self.dataset_yaml.parent / "Detection")
        dataset_results = validator.validate_detection_dataset(self.dataset_yaml)
        
        if not dataset_results['valid']:
            print("-  Dataset validation failed")
            for error in dataset_results['errors']:
                print(f"  {error}")
            raise ValueError("Dataset validation failed")
        
        print("-  Environment validation completed")
        return dataset_results
    
    def initialize_model(self):
        print(f"-  Initializing {self.base_model} detection model...")
        
        try:
            # Load base model
            self.model = YOLO(self.base_model)
            print(f"-  Loaded base model: {self.base_model}")
            
            # Print model info
            model_info = self.model.info()
            if model_info:
                print(f"Model parameters: {model_info}")
            
        except Exception as e:
            print(f"-  Failed to initialize model: {e}")
            raise
    
    def train(self, resume: bool = False) -> Dict:
        print("-  Starting detection model training...")
        
        if self.model is None:
            self.initialize_model()
        
        # Generate dynamic model name
        model_name = self.generate_model_name()
        
        # Prepare training arguments
        train_args = {
            'data': str(self.dataset_yaml),
            'epochs': self.epochs,
            'imgsz': self.image_size,
            'batch': self.batch_size,
            'lr0': self.learning_rate,
            'device': self.device,
            'workers': self.workers,
            'patience': self.patience,
            'project': str(self.results_dir),
            'name': model_name,
            'exist_ok': True,
            'pretrained': True,
            'verbose': True,
            'seed': 42,
            'resume': resume,
        }
        
        print("-  Training configuration:")
        for key, value in train_args.items():
            print(f"  {key}: {value}")
        
        try:
            # Start training
            start_time = time.time()
            print("⏱️  Training started...")
            
            self.training_results = self.model.train(**train_args)
            
            training_time = time.time() - start_time
            print(f"-  Training completed in {training_time:.2f} seconds")
            
            # Save the best model
            self.save_best_model()
            
            # Generate training report
            training_report = self.generate_training_report(training_time)
            
            return training_report
            
        except Exception as e:
            print(f"-  Training failed: {e}")
            raise
    
    def save_best_model(self):
        try:
            # Get the best model path from results
            best_model_path = self.training_results.save_dir / "weights" / "best.pt"
            
            if best_model_path.exists():
                # Generate dynamic model name
                model_name = self.generate_model_name()
                saved_model_path = self.models_dir / f"{model_name}.pt"
                
                import shutil
                shutil.copy2(best_model_path, saved_model_path)
                
                print(f"-  Best model saved to: {saved_model_path}")
            else:
                print("-   Best model not found in results")
                
        except Exception as e:
            print(f"-  Failed to save best model: {e}")
    
    def generate_training_report(self, training_time: float) -> Dict:
        print("-  Generating training report...")
        
        try:
            # Load training results
            results_csv = self.training_results.save_dir / "results.csv"
            
            if results_csv.exists():
                results_df = pd.read_csv(results_csv)
                
                # Get final metrics
                final_metrics = results_df.iloc[-1]
                best_epoch = results_df['metrics/mAP50(B)'].idxmax()
                best_metrics = results_df.iloc[best_epoch]
                
                report = {
                    'training_time': training_time,
                    'total_epochs': len(results_df),
                    'best_epoch': best_epoch + 1,
                    'final_metrics': {
                        'mAP50': float(final_metrics.get('metrics/mAP50(B)', 0)),
                        'mAP50-95': float(final_metrics.get('metrics/mAP50-95(B)', 0)),
                        'precision': float(final_metrics.get('metrics/precision(B)', 0)),
                        'recall': float(final_metrics.get('metrics/recall(B)', 0)),
                        'train_loss': float(final_metrics.get('train/box_loss', 0)),
                        'val_loss': float(final_metrics.get('val/box_loss', 0))
                    },
                    'best_metrics': {
                        'mAP50': float(best_metrics.get('metrics/mAP50(B)', 0)),
                        'mAP50-95': float(best_metrics.get('metrics/mAP50-95(B)', 0)),
                        'precision': float(best_metrics.get('metrics/precision(B)', 0)),
                        'recall': float(best_metrics.get('metrics/recall(B)', 0))
                    }
                }
                
                # Create visualizations
                self.create_training_plots(results_df)
                
                # Print summary
                print("📈 Training Summary:")
                print(f"  Total epochs: {report['total_epochs']}")
                print(f"  Best epoch: {report['best_epoch']}")
                print(f"  Best mAP@0.5: {report['best_metrics']['mAP50']:.3f}")
                print(f"  Best mAP@0.5:0.95: {report['best_metrics']['mAP50-95']:.3f}")
                print(f"  Training time: {self.format_time(training_time)}")
                
                return report
                
            else:
                print("-   Results CSV not found, generating basic report")
                return {
                    'training_time': training_time,
                    'status': 'completed_no_detailed_metrics'
                }
                
        except Exception as e:
            print(f"-  Failed to generate training report: {e}")
            return {
                'training_time': training_time,
                'status': 'completed_with_errors',
                'error': str(e)
            }
    
    def create_training_plots(self, results_df: pd.DataFrame):
        try:
            print("-  Creating training visualization plots...")
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Detection Model Training Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Loss curves
            axes[0, 0].plot(results_df.index, results_df['train/box_loss'], label='Train Box Loss', linewidth=2)
            axes[0, 0].plot(results_df.index, results_df['val/box_loss'], label='Val Box Loss', linewidth=2)
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: mAP curves
            axes[0, 1].plot(results_df.index, results_df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2, color='green')
            axes[0, 1].plot(results_df.index, results_df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2, color='blue')
            axes[0, 1].set_title('mAP Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Precision and Recall
            axes[1, 0].plot(results_df.index, results_df['metrics/precision(B)'], label='Precision', linewidth=2, color='orange')
            axes[1, 0].plot(results_df.index, results_df['metrics/recall(B)'], label='Recall', linewidth=2, color='red')
            axes[1, 0].set_title('Precision & Recall')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Score')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Learning Rate
            if 'lr/pg0' in results_df.columns:
                axes[1, 1].plot(results_df.index, results_df['lr/pg0'], label='Learning Rate', linewidth=2, color='purple')
                axes[1, 1].set_title('Learning Rate Schedule')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Learning Rate')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            else:
                axes[1, 1].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Learning Rate Schedule')
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = self.results_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = plots_dir / f"detection_training_plots_{timestamp}.png"
            
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"-  Training plots saved to: {plot_path}")
            
        except Exception as e:
            print(f"-  Failed to create training plots: {e}")
    
    @staticmethod
    def format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


def main():
    print("-  Starting Detection Model Training")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = DetectionTrainer()
        
        # Check for resume option
        resume = len(sys.argv) > 1 and sys.argv[1] == '--resume'
        if resume:
            print("Resuming training from last checkpoint...")
        
        # Start training
        results = trainer.train(resume=resume)
        
        print("\n-  Training completed successfully!")
        print("-  Final Results:")
        if 'best_metrics' in results:
            for metric, value in results['best_metrics'].items():
                print(f"  {metric}: {value:.3f}")
        
        model_name = trainer.generate_model_name()
        print(f"\n-  Best model saved to: models/{model_name}.pt")
        print(f"📁 Training results in: {trainer.results_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\n-  Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 