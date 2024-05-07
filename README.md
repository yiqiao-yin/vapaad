# 🚀 Vision Augmentation Prediction Autoencoder with Attention Design (VAPAAD)

This repository accompanies the paper "Vision Augmentation Prediction Autoencoder with Attention Design (VAPAAD)". The main code is located in the `src` folder, with the primary script being `vapaad.py`. The repository also includes a folder `gif` containing animations and a `requirements.txt` file specifying the necessary dependencies.

## 🔍 Overview

VAPAAD is a sophisticated neural network architecture designed for video processing tasks such as video frame prediction and unsupervised learning. The model leverages a dual-encoder structure and integrates data augmentation directly into the video processing pipeline. It uses self-attention mechanisms to capture long-range dependencies within video sequences, enhancing its predictive capabilities.

## 📁 Repository Structure

- 📂 `src/`: Contains the source code, including the main script `vapaad.py`.
- 📂 `gif/`: Contains animation files.
- 📄 `requirements.txt`: Lists the necessary Python packages.
- 📄 `demo.ipynb`: A demo notebook to illustrate the usage of the model.

## 🛠️ Requirements

The dependencies for this project are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Virtual Environment Setup

To set up a virtual environment and install the dependencies, follow these steps:

1. **Create a virtual environment**:

    ```bash
    python3 -m venv vapaad-env
    ```

2. **Activate the virtual environment**:

    - On Windows:
    ```bash
    vapaad-env\\Scripts\\activate
    ```
    
    - On macOS and Linux:
    ```bash
    source vapaad-env/bin/activate
    ```

3. **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

## Running the Code

To run the code, you can open and execute `demo.ipynb` using Jupyter Notebook or JupyterLab. This notebook demonstrates how to utilize the VAPAAD model for video processing tasks.

### Usage Example

```py
# Initialize a new VAPAAD model
vapaad_model = VAPAAD(input_shape=(19, 64, 64, 1))

# Assuming x_train and y_train are already defined and loaded
num_samples = 64
indices = np.random.choice(x_train.shape[0], num_samples, replace=True)
x_train_sub = x_train[indices]
y_train_sub = y_train[indices]

# Run the model
BATCH_SIZE = 3
if tf.test.gpu_device_name() != '':
    with tf.device('/device:GPU:0'):
        vapaad_model.train(x_train_sub, y_train_sub, batch_size=BATCH_SIZE)
else:
        vapaad_model.train(x_train_sub, y_train_sub, batch_size=BATCH_SIZE)
```

## Author

The VAPAAD model is developed by Yiqiao Yin, who can be reached at eagle0504@gmail.com for further inquiries or support related to this implementation.
