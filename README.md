# 🚀 Vision Augmentation Prediction Autoencoder with Attention Design (VAPAAD)

This repository accompanies the paper "Vision Augmentation Prediction Autoencoder with Attention Design (VAPAAD)". The main code is located in the `src` folder, with the primary script being `vapaad.py`. The repository also includes a folder `gifs` containing animations and a `requirements.txt` file specifying the necessary dependencies.

## 🔍 Overview

VAPAAD is a sophisticated neural network architecture designed for video processing tasks such as video frame prediction and unsupervised learning. The model leverages a dual-encoder structure and integrates data augmentation directly into the video processing pipeline. It uses self-attention mechanisms to capture long-range dependencies within video sequences, enhancing its predictive capabilities.

## 🚀 Quick Start

```bash
# 1. Initialize project and add dependencies
uv init my_vapaad_proj && cd my_vapaad_proj
uv add numpy tensorflow keras matplotlib imageio pillow ipython jupyter ipywidgets

# 2. Clone and run VAPAAD
git clone https://github.com/yiqiao-yin/vapaad.git
cd vapaad/test
uv run python test_vapaad.py

# 3. Check results in test_results/ directory
```

**That's it!** The script will download data, train the model, and generate visualizations automatically.

## 📁 Repository Structure

```
VAPAAD/
|
|___.gitignore
|___.pylintrc
|___LICENSE
|___README.md
|___requirements.txt
|
|___gifs/
|   |___[animation files]
|
|___src/
|   |___acquire_data.py
|   |___plot_images.py
|   |___README.md
|   |___vapaad.py
|
|___test/
    |___demo.ipynb
    |___README.md
    |___test_vapaad.py
```

## 🛠️ Installation & Setup

We recommend using [uv](https://docs.astral.sh/uv/) for fast and reliable Python package management. Follow these step-by-step instructions:

### Step 1: Create a new project with uv

```bash
uv init my_vapaad_proj
cd my_vapaad_proj
```

### Step 2: Add required packages

```bash
# Add all dependencies in one command
uv add numpy tensorflow keras matplotlib imageio pillow ipython jupyter ipywidgets
```

### Step 3: Clone and setup VAPAAD

```bash
# Clone the repository
git clone https://github.com/yiqiao-yin/vapaad.git
cd vapaad

# Or if you want to copy the source files to your project
cp -r vapaad/src ./
cp -r vapaad/test ./
```

### Alternative: Traditional pip installation

If you prefer using pip with virtual environments:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Option 1: Run the comprehensive test suite

```bash
# Navigate to test directory
cd test

# Run the complete VAPAAD pipeline (CPU-only by default)
uv run python test_vapaad.py

# Or specify device preference:
VAPAAD_DEVICE=cpu uv run python test_vapaad.py     # Force CPU-only
VAPAAD_DEVICE=gpu uv run python test_vapaad.py     # Prefer GPU, fallback to CPU
VAPAAD_DEVICE=auto uv run python test_vapaad.py    # Auto-detect best device
```

**Device Options:**
- `cpu` (default): Force CPU-only execution for maximum compatibility
- `gpu`: Prefer GPU with automatic CPU fallback if GPU fails
- `auto`: Let TensorFlow automatically choose the best available device

This will automatically:
- Download the Moving MNIST dataset
- Train the VAPAAD model on your chosen device
- Generate predictions and visualizations  
- Create GIF animations
- Save all results to `test_results/` directory

### Option 2: Interactive Jupyter notebook

```bash
# Start Jupyter in the test directory
cd test
jupyter notebook demo.ipynb
```

### Option 3: Programmatic usage

```python
from src.vapaad import VAPAAD
import numpy as np
import tensorflow as tf

# Initialize a new VAPAAD model
vapaad_model = VAPAAD(input_shape=(19, 64, 64, 1))

# Assuming x_train and y_train are already defined and loaded
num_samples = 64
indices = np.random.choice(x_train.shape[0], num_samples, replace=True)
x_train_sub = x_train[indices]
y_train_sub = y_train[indices]

# Train the model
BATCH_SIZE = 3
if tf.test.gpu_device_name() != '':
    with tf.device('/device:GPU:0'):
        vapaad_model.train(x_train_sub, y_train_sub, batch_size=BATCH_SIZE)
else:
    vapaad_model.train(x_train_sub, y_train_sub, batch_size=BATCH_SIZE)
```

## 📊 Output & Results

After running the test suite, you'll find comprehensive results in the `test_results/` directory:

```
test_results/
|___test_results.json              # Complete metrics and timing data
|___sample_frames.png              # Sample input data visualization
|___sequence_visualization.png     # Input/output sequence comparison  
|___prediction_visualization.png   # Model predictions vs ground truth
|___predicted_gifs/                # Generated video animations
    |___example_0_original.gif
    |___example_0_predicted.gif
    |___...
```

## ⚙️ Device Configuration & Performance

VAPAAD supports flexible CPU/GPU execution with intelligent device selection:

### **Device Options**
| Setting | Behavior | Best For |
|---------|----------|----------|
| `cpu` (default) | Force CPU-only execution | Maximum compatibility, any system |
| `gpu` | Prefer GPU, fallback to CPU | Systems with proper CUDA/cuDNN setup |
| `auto` | Auto-detect best device | Let TensorFlow choose optimally |

### **Performance Comparison**
- **CPU Mode**: Universal compatibility, slower training (~10-15 min)
- **GPU Mode**: 5-10x faster training (~2-3 min), requires GPU setup
- **Memory Usage**: ~4-6GB RAM (CPU) or ~2-4GB VRAM (GPU)

### **GPU Setup Requirements**
For `VAPAAD_DEVICE=gpu` mode, ensure you have:
```bash
# Install GPU-enabled TensorFlow
uv add tensorflow[gpu]  # or tensorflow-gpu

# System requirements:
# - NVIDIA GPU with CUDA support
# - CUDA Toolkit (compatible version)
# - cuDNN library properly installed
```

### **Troubleshooting**
- **"No DNN in stream executor"**: Use `VAPAAD_DEVICE=cpu` mode
- **Out of memory errors**: Reduce batch size or use CPU mode
- **Slow training**: Consider GPU mode if hardware supports it

## 🔧 Development & Testing

The project includes comprehensive testing infrastructure:

- **Pylint compliance**: Code follows PEP 8 standards with type hints
- **Device flexibility**: CPU/GPU execution with automatic fallback
- **Automated testing**: Complete pipeline testing with `test_vapaad.py`
- **JSON logging**: Structured result saving for analysis
- **Visualization**: Automated plot and GIF generation

## 📚 Project Structure Details

|___**src/**: Core implementation
|   |___`vapaad.py`: Main VAPAAD model implementation
|   |___`acquire_data.py`: Data loading and preprocessing
|   |___`plot_images.py`: Visualization utilities
|
|___**test/**: Testing and demonstration
|   |___`test_vapaad.py`: Comprehensive test suite
|   |___`demo.ipynb`: Interactive Jupyter notebook
|   |___`README.md`: Testing documentation
|
|___**gifs/**: Sample output animations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch using `uv`
3. Follow the existing code style (pylint compliant)
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 👤 Author

The VAPAAD model is developed by **Yiqiao Yin**.
- 📧 Contact: eagle0504@gmail.com
- 🔗 For inquiries or support related to this implementation

## 🙏 Citation

If you use VAPAAD in your research, please cite the accompanying paper:

```bibtex
@article{vapaad2024,
  title={Vision Augmentation Prediction Autoencoder with Attention Design (VAPAAD)},
  author={Yin, Yiqiao},
  journal={TBD},
  year={2024}
}
```
