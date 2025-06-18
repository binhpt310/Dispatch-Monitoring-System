import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import shutil

import torch
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from config.training_config import ClassificationConfig
from utils.data_utils import DatasetValidator, DataPreprocessor


class ClassificationTrainer:   
    def __init__(self):
        self.model = None
        self.training_results = None
        self.class_names = []
        
        # Device detection and configuration
        self.device = self._detect_device()
        print(f"-  Training device: {self.device}")
        
        # Configuration with CPU/GPU optimization
        self.epochs = 100
        self.learning_rate = 0.001
        self.patience = 50
        
        # Optimize for device
        if self.device == 'cpu':
            # CPU optimized settings
            self.batch_size = 8  # Smaller batch for CPU
            self.image_size = 224  # Standard classification image size
            self.workers = min(4, os.cpu_count())  # Limit workers on CPU
            print("-   CPU mode: Using reduced batch size for better performance")
        else:
            # GPU optimized settings
            self.batch_size = 16
            self.image_size = 224  # Standard classification image size
            self.workers = 8
        
        # Paths
        self.dataset_root = Path("Dataset/classification")
        self.results_dir = Path("results")
        self.models_dir = Path("models")
        self.base_model = "yolo11m-cls.pt"
        
        self.validate_environment()
        self.prepare_dataset()
    
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
        
        # Create dynamic name: classification_model_{base_name}_best_{epochs}_{device}_{date}
        model_name = f"classification_model_{base_name}_best_{self.epochs}_{device_suffix}_{current_date}"
        
        return model_name
    

    def validate_environment(self):
        print("-  Validating classification training environment...")
        
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
        if not self.dataset_root.exists():
            print(f"-  Dataset root not found: {self.dataset_root}")
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")
        
        # Validate classification dataset
        validator = DatasetValidator(self.dataset_root)
        dataset_results = validator.validate_classification_dataset()
        
        if not dataset_results['valid']:
            print("-  Dataset validation failed")
            for error in dataset_results['errors']:
                print(f"  {error}")
            raise ValueError("Dataset validation failed")
        
        print("-  Environment validation completed")
        return dataset_results
    
    def prepare_dataset(self):
        print("📂 Preparing classification dataset...")
        
        # Create structured dataset directory for YOLO classification
        yolo_dataset_dir = self.results_dir / "classification_dataset"
        
        # Collect all classes from both categories
        all_classes = set()
        
        for category in ['dish', 'tray']:
            category_path = self.dataset_root / category
            if category_path.exists():
                for class_dir in category_path.iterdir():
                    if class_dir.is_dir():
                        # Create combined class name: category_state
                        combined_class = f"{category}_{class_dir.name}"
                        all_classes.add(combined_class)
        
        self.class_names = sorted(list(all_classes))
        print(f"🏷️  Detected classes: {self.class_names}")
        
        # Create train/val splits for YOLO format
        for split in ['train', 'val']:
            split_dir = yolo_dataset_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for combined_class in self.class_names:
                class_dir = split_dir / combined_class
                class_dir.mkdir(exist_ok=True)
        
        # Copy and split images
        split_ratio = 0.8  # 80% train, 20% val
        
        for category in ['dish', 'tray']:
            category_path = self.dataset_root / category
            if not category_path.exists():
                continue
                
            for class_dir in category_path.iterdir():
                if not class_dir.is_dir():
                    continue
                
                combined_class = f"{category}_{class_dir.name}"
                images = list(class_dir.glob('*'))
                
                # Shuffle and split
                np.random.seed(42)  # For reproducibility
                np.random.shuffle(images)
                
                split_idx = int(len(images) * split_ratio)
                train_images = images[:split_idx]
                val_images = images[split_idx:]
                
                # Copy files
                for img_list, split in [(train_images, 'train'), (val_images, 'val')]:
                    target_dir = yolo_dataset_dir / split / combined_class
                    for img_path in img_list:
                        shutil.copy2(img_path, target_dir / img_path.name)
                
                print(f"  {combined_class}: {len(train_images)} train, {len(val_images)} val")
        
        # Update dataset path
        self.prepared_dataset_path = yolo_dataset_dir
        print(f"-  Dataset prepared at: {self.prepared_dataset_path}")
    
    def initialize_model(self):
        print(f"-  Initializing {self.base_model} classification model...")
        
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
        print("-  Starting classification model training...")
        
        if self.model is None:
            self.initialize_model()
        
        # Generate dynamic model name
        model_name = self.generate_model_name()
        
        # Prepare training arguments
        train_args = {
            'data': str(self.prepared_dataset_path),
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
                # Copy to configured location
                self.models_dir.mkdir(exist_ok=True)
                
                # Generate dynamic model name
                model_name = self.generate_model_name()
                saved_model_path = self.models_dir / f"{model_name}.pt"
                
                shutil.copy2(best_model_path, saved_model_path)
                
                print(f"-  Best model saved to: {saved_model_path}")
            else:
                print("-   Best model not found in results")
                
        except Exception as e:
            print(f"-  Failed to save best model: {e}")
    
    def generate_training_report(self, training_time: float) -> Dict:
        print("-  Generating classification training report...")
        try:
            # Load training results
            results_file = self.training_results.save_dir / "results.csv"
            
            if results_file.exists():
                results_df = pd.read_csv(results_file)
                results_df.columns = results_df.columns.str.strip()  # Remove whitespace
                
                # Calculate metrics
                final_metrics = results_df.iloc[-1]
                best_epoch = results_df['metrics/accuracy_top1'].idxmax()
                best_metrics = results_df.iloc[best_epoch]
                
                report = {
                    'training_time': training_time,
                    'total_epochs': len(results_df),
                    'best_epoch': best_epoch + 1,
                    'final_metrics': {
                        'accuracy_top1': float(final_metrics.get('metrics/accuracy_top1', 0)),
                        'accuracy_top5': float(final_metrics.get('metrics/accuracy_top5', 0)),
                        'train_loss': float(final_metrics.get('train/loss', 0)),
                        'val_loss': float(final_metrics.get('val/loss', 0))
                    },
                    'best_metrics': {
                        'accuracy_top1': float(best_metrics.get('metrics/accuracy_top1', 0)),
                        'accuracy_top5': float(best_metrics.get('metrics/accuracy_top5', 0))
                    }
                }
                
                # Print summary
                print("📈 Classification Training Summary:")
                print(f"  Training time: {self.format_time(training_time)}")
                print(f"  Total epochs: {report['total_epochs']}")
                print(f"  Best epoch: {report['best_epoch']}")
                print(f"  Best accuracy: {report['best_metrics']['accuracy_top1']:.3f}")
                
                # Create visualizations
                self.create_training_plots(results_df)
                
                # Validate model if possible
                try:
                    validation_metrics = self._validate_trained_model()
                    report['validation_metrics'] = validation_metrics
                except Exception as e:
                    print(f"-   Could not run validation: {e}")
                
                return report
                
            else:
                print("-   Results file not found, generating basic report")
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
    
    def _validate_trained_model(self) -> Dict:
        print("🧪 Validating trained model...")
        
        try:
            # Load the trained model
            trained_model = YOLO(str(self.prepared_dataset_path))
            
            # Run validation with standardized output location
            val_results = trained_model.val(
                data=str(self.prepared_dataset_path),
                project=str(self.results_dir.parent),
                name=self.results_dir.name,
                exist_ok=True
            )
            
            validation_metrics = {
                'accuracy_top1': float(val_results.top1),
                'accuracy_top5': float(val_results.top5) if hasattr(val_results, 'top5') else 0.0,
            }
            
            print(f"-  Validation completed - Top1: {validation_metrics['accuracy_top1']:.4f}")
            
            return validation_metrics
            
        except Exception as e:
            print(f"-  Validation failed: {e}")
            return {'error': str(e)}
    
    def create_training_plots(self, results_df: pd.DataFrame):
        try:
            print("-  Creating classification training visualization plots...")
            
            # Set style
            plt.style.use('seaborn-v0_8')
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Classification Model Training Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Loss curves
            axes[0, 0].plot(results_df.index, results_df['train/loss'], label='Train Loss', linewidth=2, color='blue')
            axes[0, 0].plot(results_df.index, results_df['val/loss'], label='Val Loss', linewidth=2, color='red')
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Accuracy curves
            axes[0, 1].plot(results_df.index, results_df['metrics/accuracy_top1'], 
                           label='Top-1 Accuracy', linewidth=2, color='green')
            
            if 'metrics/accuracy_top5' in results_df.columns:
                axes[0, 1].plot(results_df.index, results_df['metrics/accuracy_top5'], 
                               label='Top-5 Accuracy', linewidth=2, color='orange')
            
            axes[0, 1].set_title('Accuracy Curves')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Learning Rate
            if 'lr/pg0' in results_df.columns:
                axes[1, 0].plot(results_df.index, results_df['lr/pg0'], 
                               label='Learning Rate', linewidth=2, color='purple')
                axes[1, 0].set_title('Learning Rate Schedule')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Learning Rate')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Learning Rate Schedule')
            
            # Plot 4: Class Distribution (if available)
            try:
                class_counts = []
                for class_name in self.class_names:
                    val_dir = self.prepared_dataset_path / 'val' / class_name
                    if val_dir.exists():
                        count = len(list(val_dir.glob('*')))
                        class_counts.append(count)
                    else:
                        class_counts.append(0)
                
                if class_counts:
                    axes[1, 1].bar(range(len(self.class_names)), class_counts, 
                                  color='skyblue', alpha=0.7)
                    axes[1, 1].set_title('Validation Set Class Distribution')
                    axes[1, 1].set_xlabel('Class')
                    axes[1, 1].set_ylabel('Number of Samples')
                    axes[1, 1].set_xticks(range(len(self.class_names)))
                    axes[1, 1].set_xticklabels(self.class_names, rotation=45, ha='right')
                    axes[1, 1].grid(True, alpha=0.3)
                else:
                    axes[1, 1].text(0.5, 0.5, 'Class Distribution\nData Not Available', 
                                   ha='center', va='center', transform=axes[1, 1].transAxes)
                    axes[1, 1].set_title('Class Distribution')
                    
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Class Distribution\nError: {str(e)[:30]}...', 
                               ha='center', va='center', transform=axes[1, 1].transAxes)
                axes[1, 1].set_title('Class Distribution')
            
            plt.tight_layout()
            
            # Save plot
            plot_path = self.results_dir / 'training_plots.png'
            plot_path.parent.mkdir(exist_ok=True)
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
    print("-  Starting Classification Model Training")
    print("=" * 50)
    
    try:
        trainer = ClassificationTrainer()
        
        resume = len(sys.argv) > 1 and sys.argv[1] == '--resume'
        if resume:
            print("Resuming training from last checkpoint...")
        
        results = trainer.train(resume=resume)
        
        print("\n-  Training completed successfully!")
        print("-  Final Results:")
        if 'best_metrics' in results:
            for metric, value in results['best_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        if 'num_classes' in results:
            print(f"  Number of classes: {results['num_classes']}")
            print(f"  Classes: {', '.join(results['class_names'])}")
        
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