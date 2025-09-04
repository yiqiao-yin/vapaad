#!/usr/bin/env python3
"""
Test script for VAPAAD model training and evaluation.

This script demonstrates the full pipeline of the VAPAAD (Vision Augmentation 
Prediction Autoencoder with Attention Design) model, including data acquisition,
model training, prediction, visualization, and GIF generation.

Copyright © 2010-2024 Present Yiqiao Yin
"""

import json
import os
import sys
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import imageio

# Handle display imports gracefully for non-interactive environments
try:
    from IPython.display import Image, display
    from ipywidgets import widgets, HBox
    INTERACTIVE_AVAILABLE = True
except ImportError:
    INTERACTIVE_AVAILABLE = False

# Use non-interactive backend for matplotlib in testing environments
matplotlib.use('Agg')

# Configure TensorFlow execution mode
# Users can set VAPAAD_DEVICE environment variable: 'cpu', 'gpu', or 'auto' (default)
DEVICE_PREFERENCE = os.environ.get('VAPAAD_DEVICE', 'cpu').lower()

if DEVICE_PREFERENCE == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU-only execution
    print("🖥️  Device preference: CPU-only (set VAPAAD_DEVICE=gpu or VAPAAD_DEVICE=auto to change)")
elif DEVICE_PREFERENCE == 'gpu':
    # Allow GPU usage, remove CPU-only restriction
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
        del os.environ['CUDA_VISIBLE_DEVICES']
    print("🚀 Device preference: GPU-preferred (will fallback to CPU if GPU unavailable)")
elif DEVICE_PREFERENCE == 'auto':
    # Let TensorFlow decide automatically
    if 'CUDA_VISIBLE_DEVICES' in os.environ and os.environ['CUDA_VISIBLE_DEVICES'] == '-1':
        del os.environ['CUDA_VISIBLE_DEVICES']
    print("⚡ Device preference: Auto-detect (TensorFlow will choose best available)")
else:
    print(f"⚠️  Unknown VAPAAD_DEVICE value: {DEVICE_PREFERENCE}. Using CPU-only as fallback.")
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Reduce TF logging

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.plot_images import plot_image_sequences, rescale_and_discretize
from src.vapaad import VAPAAD


def sequence_ce_sum(y_true, y_pred):
    """Paper-style summed cross-entropy per sequence."""
    # Always cast to float32 to avoid mixed precision issues
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    # Get the actual number of dimensions to handle both 4D and 5D cases
    ndims = len(bce.shape)
    if ndims == 4:  # (batch, time, height, width)
        ce_sum_per_seq = tf.reduce_sum(bce, axis=[1, 2, 3])
    else:  # (batch, time, height, width, channels)
        ce_sum_per_seq = tf.reduce_sum(bce, axis=[1, 2, 3, 4])
    return tf.reduce_mean(ce_sum_per_seq)


def mse_seq(y_true, y_pred):
    """MSE per sequence."""
    # Always cast to float32 to avoid mixed precision issues
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    se = tf.square(y_pred - y_true)
    # Get the actual number of dimensions
    ndims = len(se.shape)
    if ndims == 4:  # (batch, time, height, width)
        mse_per_seq = tf.reduce_mean(se, axis=[1, 2, 3])
    else:  # (batch, time, height, width, channels)
        mse_per_seq = tf.reduce_mean(se, axis=[1, 2, 3, 4])
    return tf.reduce_mean(mse_per_seq)


class VAPAADTester:
    """
    A comprehensive tester class for the VAPAAD model that handles data loading,
    model training, evaluation, and result visualization.
    """
    
    def __init__(self, output_dir: str = "test_results") -> None:
        """
        Initialize the VAPAAD tester.
        
        Args:
            output_dir: Directory to save test results and outputs
        """
        self.output_dir = output_dir
        self.results: Dict[str, Any] = {}
        self.model: Optional[Any] = None
        self.x_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.x_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_val_pred: Optional[np.ndarray] = None
        self.train_dataset: Optional[np.ndarray] = None
        self.val_dataset: Optional[np.ndarray] = None
        
        # Store device preference
        self.device_preference = DEVICE_PREFERENCE
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
    
    def _get_device_context(self) -> str:
        """
        Determine the appropriate device context based on user preference and availability.
        
        Returns:
            Device string for tf.device() context
        """
        if self.device_preference == 'cpu':
            return '/device:CPU:0'
        elif self.device_preference == 'gpu':
            if tf.test.gpu_device_name():
                return '/device:GPU:0'
            else:
                print("⚠️  GPU requested but not available. Falling back to CPU.")
                return '/device:CPU:0'
        elif self.device_preference == 'auto':
            # Let TensorFlow choose, but provide context for logging
            return '/device:GPU:0' if tf.test.gpu_device_name() else '/device:CPU:0'
        else:
            return '/device:CPU:0'
    
    def _should_use_gpu(self) -> bool:
        """Check if GPU should be used based on preference and availability."""
        if self.device_preference == 'cpu':
            return False
        return tf.test.gpu_device_name() != ''
    
    def acquire_data(self) -> None:
        """
        Download and preprocess the Moving MNIST dataset.
        """
        print("Acquiring data...")
        start_time = time.time()
        
        # Download and load the dataset
        fpath = keras.utils.get_file(
            "moving_mnist.npy",
            "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy",
        )
        dataset = np.load(fpath)
        
        # Swap axes and add channel dimension
        dataset = np.swapaxes(dataset, 0, 1)
        dataset = np.expand_dims(dataset, axis=-1)
        
        # Split into train and validation sets
        indexes = np.arange(dataset.shape[0])
        np.random.shuffle(indexes)
        train_index = indexes[:int(0.9 * dataset.shape[0])]
        val_index = indexes[int(0.9 * dataset.shape[0]):]
        train_dataset = dataset[train_index]
        val_dataset = dataset[val_index]
        
        # Normalize data
        train_dataset = train_dataset / 255
        val_dataset = val_dataset / 255
        
        # Create shifted frames
        self.x_train, self.y_train = self._create_shifted_frames(train_dataset)
        self.x_val, self.y_val = self._create_shifted_frames(val_dataset)
        
        # Store datasets for later use
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Convert to float16 to reduce memory usage
        self.x_train = self.x_train.astype(np.float16)
        self.y_train = self.y_train.astype(np.float16)
        
        data_time = time.time() - start_time
        
        # Store results
        self.results['data_acquisition'] = {
            'processing_time': data_time,
            'train_shape': list(self.x_train.shape),
            'val_shape': list(self.x_val.shape),
            'data_type': str(self.x_train.dtype)
        }
        
        print(f"Training Dataset Shapes: {self.x_train.shape}, {self.y_train.shape}")
        print(f"Validation Dataset Shapes: {self.x_val.shape}, {self.y_val.shape}")
        print(f"Data acquisition completed in {data_time:.2f} seconds")
    
    def _create_shifted_frames(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create shifted frames where x is frames 0 to n-1, and y is frames 1 to n.
        
        Args:
            data: Input video data
            
        Returns:
            Tuple of (x_frames, y_frames)
        """
        x_frames = data[:, 0:data.shape[1] - 1, :, :]
        y_frames = data[:, 1:data.shape[1], :, :]
        return x_frames, y_frames
    
    def visualize_sample_data(self) -> None:
        """
        Create and save visualization of sample training data.
        """
        print("Creating sample data visualization...")
        
        # Create visualization figure
        fig, axes = plt.subplots(4, 5, figsize=(10, 8))
        
        # Plot sequential images for one random example
        data_choice = np.random.choice(range(len(self.train_dataset)), size=1)[0]
        for idx, axis in enumerate(axes.flat):
            axis.imshow(np.squeeze(self.train_dataset[data_choice][idx]), cmap="gray")
            axis.set_title(f"Frame {idx + 1}")
            axis.axis("off")
        
        plt.suptitle(f"Sample frames for example {data_choice}")
        plt.tight_layout()
        
        # Save figure
        sample_plot_path = os.path.join(self.output_dir, "sample_frames.png")
        plt.savefig(sample_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Create sequence visualization
        self._create_sequence_visualization()
        
        self.results['visualization'] = {
            'sample_plot_saved': sample_plot_path,
            'sample_example_index': int(data_choice)
        }
        
        print(f"Sample data visualization saved to {sample_plot_path}")
    
    def _create_sequence_visualization(self) -> None:
        """
        Create input/output sequence visualization.
        """
        plt.figure(figsize=(20, 6))
        plot_image_sequences(self.x_train, self.y_train, num_samples=2)
        
        sequence_plot_path = os.path.join(self.output_dir, "sequence_visualization.png")
        plt.savefig(sequence_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Sequence visualization saved to {sequence_plot_path}")
    
    def initialize_model(self) -> None:
        """
        Initialize the VAPAAD model.
        """
        print("Initializing VAPAAD model...")
        
        input_shape = (19, 64, 64, 1)
        self.model = VAPAAD(input_shape=input_shape)
        
        # Print model instructions
        self.model.__read_me__()
        
        self.results['model_initialization'] = {
            'input_shape': list(input_shape),
            'model_type': 'VAPAAD'
        }
        
        print("VAPAAD model initialized successfully")
    
    def train_model(self, num_samples: int = 64, batch_size: int = 3) -> None:
        """
        Train the VAPAAD model on a subset of data.
        
        Args:
            num_samples: Number of samples to use for training
            batch_size: Batch size for training
        """
        print(f"Training model with {num_samples} samples, batch size {batch_size}...")
        start_time = time.time()
        
        # Select random subset
        indices = np.random.choice(self.x_train.shape[0], num_samples, replace=True)
        x_train_sub = self.x_train[indices]
        y_train_sub = self.y_train[indices]
        
        print(f"Training subset shape: {x_train_sub.shape}, {y_train_sub.shape}")
        
        # Train model with user-specified device preference
        device_context = self._get_device_context()
        gpu_used = self._should_use_gpu() and 'GPU' in device_context
        
        try:
            print(f"🔥 Training on device: {device_context}")
            if gpu_used:
                print(f"📊 GPU detected: {tf.test.gpu_device_name()}")
            
            with tf.device(device_context):
                self.model.train(x_train_sub, y_train_sub, batch_size=batch_size)
                
        except Exception as device_error:
            if gpu_used:
                print(f"⚠️  GPU training failed: {device_error}")
                print("🔄 Falling back to CPU training...")
                gpu_used = False
                with tf.device('/device:CPU:0'):
                    self.model.train(x_train_sub, y_train_sub, batch_size=batch_size)
            else:
                print(f"❌ Training failed: {device_error}")
                raise
        
        training_time = time.time() - start_time
        
        self.results['training'] = {
            'num_samples': num_samples,
            'batch_size': batch_size,
            'training_time': training_time,
            'device_preference': self.device_preference,
            'device_used': device_context,
            'gpu_used': gpu_used,
            'gpu_available': tf.test.gpu_device_name() != '',
            'selected_indices': indices[:6].tolist()
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
    
    def evaluate_model(self) -> None:
        """
        Evaluate the trained model on validation data.
        """
        print("Evaluating model on validation data...")
        start_time = time.time()
        
        # Get trained generator
        trained_generator = self.model.gen_main
        
        # Make predictions on validation set with user-specified device preference
        device_context = self._get_device_context()
        
        try:
            print(f"🔮 Running predictions on device: {device_context}")
            with tf.device(device_context):
                y_val_pred = trained_generator.predict(self.x_val)
        except Exception as device_error:
            if 'GPU' in device_context:
                print(f"⚠️  GPU prediction failed: {device_error}")
                print("🔄 Falling back to CPU for predictions...")
                with tf.device('/device:CPU:0'):
                    y_val_pred = trained_generator.predict(self.x_val)
            else:
                print(f"❌ Prediction failed: {device_error}")
                raise
        
        evaluation_time = time.time() - start_time
        
        # Calculate basic metrics
        mse = np.mean((self.y_val - y_val_pred) ** 2)
        mae = np.mean(np.abs(self.y_val - y_val_pred))
        
        # Calculate custom loss functions
        print("📊 Computing custom loss metrics...")
        with tf.device(device_context):
            custom_ce_sum = sequence_ce_sum(self.y_val, y_val_pred).numpy()
            custom_mse_seq = mse_seq(self.y_val, y_val_pred).numpy()
        
        print(f"📈 Custom Metrics:")
        print(f"   • Sequence CE Sum: {custom_ce_sum:.6f}")
        print(f"   • MSE Sequence: {custom_mse_seq:.6f}")
        print(f"   • Standard MSE: {mse:.6f}")
        print(f"   • Standard MAE: {mae:.6f}")
        
        self.results['evaluation'] = {
            'true_shape': list(self.y_val.shape),
            'predicted_shape': list(y_val_pred.shape),
            'evaluation_time': evaluation_time,
            'mse': float(mse),
            'mae': float(mae),
            'custom_metrics': {
                'sequence_ce_sum': float(custom_ce_sum),
                'mse_seq': float(custom_mse_seq)
            }
        }
        
        print(f"Shape of true y_val: {self.y_val.shape}")
        print(f"Shape of predicted y_val: {y_val_pred.shape}")
        print(f"Evaluation completed in {evaluation_time:.2f} seconds")
        
        # Store predictions for visualization
        self.y_val_pred = y_val_pred
    
    def create_prediction_visualization(self) -> None:
        """
        Create and save prediction visualization comparing true vs predicted frames.
        """
        print("Creating prediction visualization...")
        
        # Set up the figure
        plt.figure(figsize=(20, 6))
        
        # Randomly select samples
        num_samples = 2
        indices = np.random.choice(self.y_val.shape[0], num_samples, replace=False)
        
        # Create visualization
        for idx, sample_index in enumerate(indices):
            # First row for true values
            for i in range(19):
                axis = plt.subplot(num_samples * 2, 19, 2 * idx * 19 + i + 1)
                plt.imshow(self.y_val[sample_index, i, :, :, 0], cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title(f'Sample {sample_index+1} - True Frames')
                plt.text(0.5, -0.1, f't={i+1}', ha='center', va='center', 
                        transform=axis.transAxes, fontsize=8)
            
            # Second row for predicted values
            for i in range(19):
                plt.subplot(num_samples * 2, 19, (2 * idx + 1) * 19 + i + 1)
                image = self.y_val_pred[sample_index, i, :, :, 0]
                result_image = rescale_and_discretize(image)
                plt.imshow(result_image, cmap='gray')
                plt.axis('off')
                if i == 0:
                    plt.title(f'Sample {sample_index+1} - Pred Frames')
        
        plt.tight_layout()
        
        # Save figure
        pred_viz_path = os.path.join(self.output_dir, "prediction_visualization.png")
        plt.savefig(pred_viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.results['prediction_visualization'] = {
            'visualization_saved': pred_viz_path,
            'samples_visualized': indices.tolist()
        }
        
        print(f"Prediction visualization saved to {pred_viz_path}")
    
    def create_prediction_gifs(self) -> None:
        """
        Create GIF animations comparing ground truth and predicted video sequences.
        """
        print("Creating prediction GIFs...")
        
        # Create directory for GIFs
        gif_dir = os.path.join(self.output_dir, "predicted_gifs")
        os.makedirs(gif_dir, exist_ok=True)
        
        # Get trained generator
        trained_generator = self.model.gen_main
        
        # Select random examples
        examples = self.val_dataset[np.random.choice(range(len(self.val_dataset)), size=5)]
        
        gif_files = []
        
        # Generate predictions and create GIFs
        for index, example in enumerate(examples):
            # Pick first/last ten frames
            frames = example[:10, ...]
            original_frames = example[10:, ...]
            new_predictions = np.zeros(shape=(10, *frames[0].shape))
            
            # Predict new set of frames
            for i in range(10):
                frames_input = example[:10 + i + 1, ...]
                new_prediction = trained_generator.predict(np.expand_dims(frames_input, axis=0))
                new_prediction = rescale_and_discretize(new_prediction)
                new_prediction = np.squeeze(new_prediction, axis=0)
                predicted_frame = np.expand_dims(new_prediction[-1, ...], axis=0)
                new_predictions[i] = predicted_frame
            
            # Create GIFs for ground truth and predictions
            for frame_set_idx, frame_set in enumerate([original_frames, new_predictions]):
                # Prepare frames for GIF
                current_frames = np.squeeze(frame_set)
                current_frames = current_frames[..., np.newaxis] * np.ones(3)
                current_frames = (current_frames * 255).astype(np.uint8)
                current_frames = list(current_frames)
                
                # Define GIF filename
                gif_type = 'original' if frame_set_idx == 0 else 'predicted'
                gif_filename = os.path.join(gif_dir, f"example_{index}_{gif_type}.gif")
                
                # Save GIF
                imageio.mimsave(gif_filename, current_frames, "GIF", duration=0.9)
                gif_files.append(gif_filename)
        
        self.results['gif_generation'] = {
            'gif_directory': gif_dir,
            'num_examples': len(examples),
            'gif_files_created': len(gif_files),
            'gif_files': gif_files
        }
        
        print(f"Created {len(gif_files)} GIF files in {gif_dir}")
    
    def save_results(self) -> None:
        """
        Save all test results to a JSON file.
        """
        # Add system information
        self.results['system_info'] = {
            'tensorflow_version': tf.__version__,
            'keras_version': keras.__version__,
            'numpy_version': np.__version__,
            'gpu_available': tf.test.gpu_device_name() != '',
            'gpu_device': tf.test.gpu_device_name()
        }
        
        # Add timestamp
        self.results['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to JSON
        results_file = os.path.join(self.output_dir, "test_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Test results saved to {results_file}")
    
    def run_full_test(self) -> None:
        """
        Run the complete test pipeline.
        """
        print("Starting full VAPAAD test pipeline...")
        pipeline_start = time.time()
        
        try:
            # Run all test steps
            self.acquire_data()
            self.visualize_sample_data()
            self.initialize_model()
            self.train_model(num_samples=64, batch_size=3)
            self.evaluate_model()
            self.create_prediction_visualization()
            self.create_prediction_gifs()
            
            # Record total time
            total_time = time.time() - pipeline_start
            self.results['pipeline'] = {
                'total_time': total_time,
                'status': 'completed'
            }
            
            print(f"\nFull test pipeline completed successfully in {total_time:.2f} seconds")
            
        except Exception as e:
            # Record error
            total_time = time.time() - pipeline_start
            self.results['pipeline'] = {
                'total_time': total_time,
                'status': 'failed',
                'error': str(e)
            }
            print(f"Test pipeline failed after {total_time:.2f} seconds: {e}")
            raise
        
        finally:
            # Always save results
            self.save_results()


def main():
    """
    Main function to run VAPAAD tests.
    """
    print("VAPAAD Model Test Suite")
    print("=" * 50)
    
    # Display device configuration info
    print(f"📱 Device Configuration:")
    print(f"   • Current preference: {DEVICE_PREFERENCE}")
    print(f"   • GPU available: {'Yes' if tf.test.gpu_device_name() else 'No'}")
    if tf.test.gpu_device_name():
        print(f"   • GPU device: {tf.test.gpu_device_name()}")
    print()
    print("💡 To change device preference, set VAPAAD_DEVICE environment variable:")
    print("   • VAPAAD_DEVICE=cpu    - Force CPU-only execution")
    print("   • VAPAAD_DEVICE=gpu    - Prefer GPU, fallback to CPU")
    print("   • VAPAAD_DEVICE=auto   - Auto-detect best device")
    print("=" * 50)
    
    # Create tester instance
    tester = VAPAADTester(output_dir="test_results")
    
    # Run full test suite
    tester.run_full_test()
    
    print("\n✅ Test suite completed. Check test_results/ directory for outputs.")


if __name__ == "__main__":
    main()