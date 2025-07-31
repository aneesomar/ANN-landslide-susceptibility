# Landslide Susceptibility Prediction using Advanced Neural Networks

A comprehensive machine learning project for predicting landslide susceptibility using advanced neural network architectures with attention mechanisms and ensemble feature selection.

## 🎯 Project Overview

This project implements an advanced landslide prediction system using a sophisticated neural network architecture that combines:
- **Ensemble feature selection** methods
- **Attention mechanisms** for feature importance weighting
- **Residual blocks** for better gradient flow
- **Focal loss** for handling class imbalance
- **Mixed precision training** for efficiency

## 🏗️ Architecture

### Model Components
- **AdvancedLandslideANN**: Main neural network with residual connections
- **AttentionLayer**: Learns feature importance weights dynamically
- **ResidualBlock**: Prevents vanishing gradients and improves training
- **FocalLoss**: Addresses class imbalance by focusing on hard examples

### Key Features
- 🔍 **Ensemble Feature Selection**: Combines statistical, tree-based, and recursive methods
- ⚖️ **Class Balancing**: Weighted sampling and focal loss for imbalanced datasets
- 🎯 **Attention Mechanism**: Dynamic feature importance learning
- 📊 **Advanced Evaluation**: Comprehensive metrics and visualizations
- 🚀 **GPU Acceleration**: CUDA support with mixed precision training
- 📈 **Threshold Optimization**: Automatic threshold tuning for best F1 score

## 📁 Project Structure

```
python/
├── modelTraining.py              # Main training script
├── train.py                      # training script for unseen data
├── normalise.py                  # Data normalization utilities
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
# Python packages
pip install pandas scikit-learn torch torchvision numpy matplotlib seaborn
```

### Required Libraries
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning utilities and metrics
- **torch**: PyTorch for neural networks
- **numpy**: Numerical computations
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

### Installation

1. Clone or download the project
2. Install dependencies:
   ```bash
   pip install -r requirements.txt  # If available
   # OR install individually:
   pip install pandas scikit-learn torch numpy matplotlib seaborn
   ```

## 📊 Data Format

### Input Data Requirements
- **Landslide data**: CSV file with geographical and environmental features
- **Non-landslide data**: CSV file with the same feature structure
- **Required columns**: All numerical features (excluding 'xcoord', 'ycoord', 'fid')
- **Coordinate columns**: 'xcoord', 'ycoord' (automatically excluded from training)

### Data Preprocessing
- Automatic conversion of boolean values to numeric (True→1, False→0)
- Missing value imputation (filled with 0)
- Robust scaling for outlier handling
- Feature selection using ensemble methods

## 🎯 Usage

### Training the Model

```bash
python modelTraining.py
```

### What the Script Does:

1. **Data Loading & Preprocessing**
   - Loads landslide and non-landslide datasets
   - Combines and shuffles data
   - Handles missing values and data type conversion

2. **Feature Selection**
   - Statistical selection (F-test)
   - Tree-based importance (Random Forest)
   - Recursive feature elimination (RFE)
   - Ensemble voting (features appearing in ≥2 methods)

3. **Model Training**
   - Advanced neural network with attention
   - Focal loss for class imbalance
   - Mixed precision training (if GPU available)
   - Learning rate scheduling with warm restarts
   - Early stopping with patience

4. **Evaluation & Analysis**
   - Comprehensive metrics (Accuracy, Precision, Recall, F1, AUC)
   - ROC curves and confusion matrices
   - Threshold optimization
   - Feature importance analysis

## 📈 Model Performance

### Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Visualizations Generated
- Training and validation loss curves
- ROC curves
- Confusion matrices
- Prediction probability distributions
- Feature importance rankings
- Threshold optimization plots

## 🔧 Configuration Options

### Model Hyperparameters
```python
# Network architecture
input_dim = X_train_scaled.shape[1]  # Auto-determined
hidden_layers = [512, 256, 128, 64]  # Layer sizes
dropout_rates = [0.4, 0.3, 0.2, 0.1]  # Dropout for each layer

# Training parameters
num_epochs = 150
batch_size = 64
learning_rate = 0.001
weight_decay = 1e-3
patience = 25  # Early stopping patience

# Focal loss parameters
alpha = 1
gamma = 2
```

### Feature Selection
```python
# Number of features to select per method
k_features = 60  # For statistical and RFE methods
min_votes = 2    # Minimum votes for ensemble selection
```

## 📊 Output Files

### Model Files
- `landslide_model_advanced_complete.pth`: Complete model package including scaler and metadata
- `best_model_advanced.pth`: Best model weights during training

### Data Files
- `X_train.csv`, `X_test.csv`: Scaled training and test features
- `y_train.csv`, `y_test.csv`: Training and test labels

### Visualization Files
- `advanced_evaluation.png`: ROC curves, confusion matrix, probability distributions
- `feature_importance_advanced.png`: Top 15 most important features
- `training_analysis.png`: Training curves and threshold optimization


## 🔬 Advanced Features

### Attention Mechanism
- Learns to focus on most relevant features
- Improves model interpretability
- Dynamic feature weighting during inference

### Residual Connections
- Prevents vanishing gradient problem
- Enables training of deeper networks
- Improves convergence speed

### Mixed Precision Training
- Automatic GPU utilization
- Faster training with maintained accuracy
- Reduced memory usage

### Ensemble Feature Selection
- Combines multiple selection methods
- Reduces overfitting to single method bias
- More robust feature subset



