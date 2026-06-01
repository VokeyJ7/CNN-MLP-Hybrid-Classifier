# ECG Arrhythmia Classification Using Deep Learning

A deep learning project that classifies electrocardiogram (ECG) signals using both Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) architectures. The system detects abnormal heart rhythms and classifies multiple arrhythmia types from ECG waveform data.

## Project Overview

This project explores two neural network approaches for ECG signal analysis:

* A Multi-Layer Perceptron (MLP) for binary ECG abnormality detection.
* A 1D Convolutional Neural Network (CNN) for multi-class arrhythmia classification.

The objective was to evaluate how different neural network architectures perform on biomedical time-series data and determine which model is more effective at learning ECG waveform patterns.

## Datasets

### PTB Diagnostic ECG Dataset

Used for binary classification:

Classes:

* Normal ECG
* Abnormal ECG

### MIT-BIH Arrhythmia Dataset

Used for multi-class heartbeat classification.

Classes:

* Normal Beat (N)
* Supraventricular Ectopic Beat (S)
* Ventricular Ectopic Beat (V)
* Fusion Beat (F)
* Unknown Beat (Q)

## MLP Classifier

The MLP model was developed for binary classification of ECG signals.

### Architecture

* Input Layer: 187 ECG features
* Hidden Layer 1
* Hidden Layer 2
* Output Layer: 2 Classes

### Training Configuration

* StandardScaler normalization
* CrossEntropyLoss
* Adam Optimizer
* Learning Rate: 0.01
* 100 Epochs

### Results

Validation Accuracy:

```text
92.31%
```

The model successfully distinguished between normal and abnormal ECG signals while providing a lightweight baseline for comparison.

## CNN Arrhythmia Classifier

The CNN model was designed specifically for ECG waveform analysis.

### Architecture

* Multiple 1D Convolutional Layers
* ReLU Activations
* Max Pooling Layers
* Adaptive Average Pooling
* Fully Connected Classification Head

### Training Configuration

* StandardScaler normalization
* PyTorch Dataset and DataLoader
* CrossEntropyLoss
* Adam Optimizer
* Learning Rate: 0.001
* 10 Epochs

### Results

Final Validation Accuracy:

```text
98.21%
```

Final Training Accuracy:

```text
98.29%
```

The CNN significantly outperformed the MLP by automatically learning local heartbeat patterns and temporal waveform features.

## Key Findings

* MLP models can effectively classify normal versus abnormal ECG signals.
* CNN architectures are more effective for arrhythmia classification because they learn waveform features directly from ECG sequences.
* Deep learning achieved high accuracy on both binary and multi-class cardiac classification tasks.
* Convolutional feature extraction improved performance by nearly 6 percentage points compared to the baseline MLP.

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib

## Author

Veltman Okey-Ejiowhor

Mechanical Engineering Student | AI/ML Researcher | Deep Learning Applications in Healthcare
