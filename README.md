# CNN + LSTM for Activity Recognition

## Overview
This project utilizes a hybrid **CNN + LSTM** architecture for **activity recognition** using time-series sensor data from the **ARAS B House dataset**. The model integrates **Convolutional Neural Networks (CNN)** for spatial feature extraction and **Long Short-Term Memory (LSTM)** networks for capturing temporal dependencies.

## Model Architecture
The CNN + LSTM architecture consists of:
- **Input Layer:** Defines the input shape of time-series data.
- **CNN Layers:**
  - **Conv1D Layer:** Extracts spatial features from input data.
  - **MaxPooling1D Layer:** Reduces dimensionality while preserving critical features.
- **LSTM Layers:**
  - **LSTM Layer:** Captures temporal dependencies in sequential data.
  - **Dropout Layer:** Prevents overfitting by randomly deactivating neurons.
- **Fully Connected Dense Layer:** Processes extracted features for classification.
- **Output Layers:**
  - Two softmax layers to classify activities for **Resident 1** and **Resident 2**.

## Key Features
- **CNN for spatial feature extraction**: Identifies localized patterns in sensor data.
- **LSTM for temporal learning**: Recognizes activity sequences over time.
- **Softmax output layers**: Enables multi-class classification.
- **Dropout and regularization**: Reduces overfitting and enhances generalization.

## Performance
| Metric       | Resident 1 | Resident 2 |
|-------------|------------|------------|
| **Accuracy** | 94.73%     | 93.44%     |
| **Precision** | 94.59%     | 94.05%     |
| **Recall**  | 95.05%     | 96.06%     |
| **F1 Score** | 94.66%     | 94.79%     |

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.x
- TensorFlow (`pip install tensorflow`)
- NumPy (`pip install numpy`)
- Pandas (`pip install pandas`)

### Steps
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/CNN-LSTM-ActivityRecognition.git
   cd CNN-LSTM-ActivityRecognition
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the training script:
   ```sh
   python train.py
   ```
