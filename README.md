# Cat vs Non-Cat Image Classifier

## Overview
This project implements a **2-layer neural network** from scratch using NumPy to classify images as containing either a cat or not. The model demonstrates core deep learning principles by processing 64x64 RGB images through a custom architecture with forward/backward propagation, activation functions, and gradient descent optimization.

---

## Key Components

### 1. **Dataset**
- **Training Set**: 209 images (64x64 RGB)
- **Test Set**: 50 images (64x64 RGB)
- **Preprocessing**:
  - Flattening: Images reshaped to vectors (12,288 features)
  - Normalization: Pixel values scaled to [0, 1] (÷255)

### 2. **Model Architecture**
- **Input Layer**: 12,288 units (64×64×3 pixels)
- **Hidden Layer**: 7 units with ReLU activation
- **Output Layer**: 1 unit with sigmoid activation (binary classification)
- **Layer Dimensions**: `(12288, 7, 1)`

### 3. **Training Configuration**
- **Loss Function**: Binary cross-entropy
- **Optimizer**: Gradient descent
- **Hyperparameters**:
  - Learning rate: 0.0075
  - Iterations: 2500
  - Batch size: Full batch

---

## Key Features
- **Core Operations Implemented**:
  - Forward/backward propagation
  - ReLU and sigmoid activations
  - Parameter initialization
  - Gradient computation
- **Modular Design**:
  - Separate utilities for activations (`my_utils.py`)
  - End-to-end training pipeline (`main.py`)
- **Efficiency**:
  - Vectorized NumPy operations
  - In-place computations
  - HDF5 data loading

---

## Performance
- **Cost Reduction**:
  - Cost after iteration 0: 0.6931
  - Cost after iteration 100: 0.6488
    ...
  - Cost after iteration 2400: 0.2873
- **Accuracy**:
  - Training: ~99%
  - Testing: ~70% (typical for this architecture/dataset)
- **Training Dynamics**:
  - Steady cost decrease indicates effective learning
  - Hidden layer size (7 units) balances under/overfitting

## Applications
- Binary image classification
- Educational resource for understanding:
  - Neural network fundamentals
  - Backpropagation mechanics
  - Activation functions
  - Foundation for more complex computer vision systems
