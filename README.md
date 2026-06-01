# Hybrid ECG Classifier: MLP and CNN Models

This project compares two deep learning approaches for ECG signal classification: a Multi-Layer Perceptron (MLP) for binary abnormal/normal detection and a 1D Convolutional Neural Network (CNN) for multi-class arrhythmia classification.

## Project Overview

The goal of this project was to classify ECG heartbeat signals using PyTorch-based neural networks. Two models were developed:

1. **MLP Classifier**

   * Classifies ECG signals as normal or abnormal.
   * Uses fully connected layers on standardized ECG feature vectors.

2. **1D CNN Classifier**

   * Classifies ECG signals into five heartbeat categories.
   * Uses convolutional layers to learn local waveform patterns from ECG sequences.

## Datasets

The project used two ECG datasets:

### PTB Diagnostic ECG Dataset

Used for binary classification:

* `ptbdb_abnormal.csv`
* `ptbdb_normal.csv`

Classes:

* Normal ECG
* Abnormal ECG

### MIT-BIH Arrhythmia Dataset

Used for multi-class heartbeat classification:

* `mitbih_train.csv`
* `mitbih_test.csv`

Output classes:

* 5 ECG heartbeat categories

## MLP Model

The MLP model uses a simple feedforward architecture:

* Input layer: 187 ECG features
* Hidden layer 1
* Hidden layer 2
* Output layer: 2 classes

The model was trained using:

* StandardScaler preprocessing
* CrossEntropyLoss
* Adam optimizer
* 100 training epochs
* Learning rate of 0.01

## MLP Results

The MLP reached:

```text
Validation Accuracy: 92.31%
```

This showed that even a simple fully connected network could learn meaningful patterns from standardized ECG signal features.

## CNN Model

The CNN model uses 1D convolutional layers to detect waveform patterns across the ECG signal sequence.

The architecture includes:

* Multiple 1D convolutional layers
* ReLU activations
* Max pooling
* Adaptive average pooling
* Flatten layer
* Fully connected classification head

The CNN was trained using:

* StandardScaler preprocessing
* Custom PyTorch Dataset and DataLoader
* CrossEntropyLoss
* Adam optimizer
* 10 training epochs
* Learning rate of 0.001

## CNN Results

The CNN model achieved strong performance over 10 epochs.

Final validation results:

```text
Validation Loss: 0.064
Validation Accuracy: 98.214%
```

The CNN outperformed the MLP because convolutional layers are better suited for detecting local signal patterns in ECG waveforms.

## Key Takeaways

* The MLP performed well for binary ECG classification.
* The CNN achieved higher accuracy on multi-class ECG classification.
* Standardization improved training stability.
* 1D convolution was effective for learning ECG waveform patterns.
* Adaptive average pooling helped simplify the transition from convolutional layers to dense layers.

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Scikit-learn
* Joblib
* Seaborn
* Matplotlib

## Author

Created by Veltman Okey-Ejiowhor as part of an AI/ML portfolio project focused on biomedical signal classification.
