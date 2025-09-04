# VAPAAD Test Suite

This directory contains the test suite and demo materials for the VAPAAD (Vision Augmentation Prediction Autoencoder with Attention Design) model.

## Contents

- `test_vapaad.py` - Main test script that runs the complete VAPAAD pipeline
- `demo.ipynb` - Jupyter notebook demonstrating VAPAAD usage (moved from root)
- `README.md` - This file

## Running Tests

### Python Script

Run the comprehensive test suite:

```bash
cd test
python test_vapaad.py
```

This will:
1. Download and preprocess the Moving MNIST dataset
2. Train a VAPAAD model on a subset of data
3. Generate predictions and visualizations
4. Create GIF animations comparing ground truth vs predictions
5. Save all results to `test_results/` directory including:
   - Training metrics and timing information
   - Sample visualizations
   - Prediction accuracy metrics
   - Generated GIF files
   - Complete results in JSON format

### Jupyter Notebook

Alternatively, run the interactive notebook:

```bash
jupyter notebook demo.ipynb
```

Note: The notebook has been modified to work in local environments (removed Google Colab dependencies).

## Output Structure

After running tests, you'll find results in `test_results/`:

```
test_results/
├── test_results.json          # Complete test results and metrics
├── sample_frames.png          # Sample input data visualization
├── sequence_visualization.png # Input/output sequence comparison
├── prediction_visualization.png # Model predictions vs ground truth
└── predicted_gifs/           # Directory with generated GIF animations
    ├── example_0_original.gif
    ├── example_0_predicted.gif
    └── ...
```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- imageio
- IPython (for notebook)
- ipywidgets (for notebook)

## Code Quality

The test script is pylint compliant and includes:
- Type hints for better code documentation
- Comprehensive error handling
- Structured logging and result saving
- Modular design for easy extension

## Copyright

Copyright © 2010-2024 Present Yiqiao Yin