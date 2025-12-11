# cost_model_LSTM: Halide Program Execution Time Prediction

![Python](https://img.shields.io/badge/Language-Python-100%25-blue)
![Architecture](https://img.shields.io/badge/Model-LSTM-orange)
![Domain](https://img.shields.io/badge/Domain-Compiler%20Optimization-green)

This repository houses a deep learning solution aimed at predicting the execution times of programs for various schedules. The primary focus is on developing a **cost model** to forecast the runtime performance of complex parallel processing tasks, specifically targeting the scheduling optimization problem often found in frameworks like **Halide**.

## 🎯 Goal and Application

The cost model provided here acts as a crucial component in **compiler optimization** and **auto-scheduling** efforts.

*   **The Problem:** Finding the fastest schedule for a program usually requires running it thousands of times, which is slow and expensive.
*   **The Solution:** This model utilizes a **Long Short-Term Memory (LSTM)** network to learn the relationship between program schedule features and execution latency.
*   **The Result:** By accurately predicting execution time without needing to run the program, the system can quickly search vast scheduling spaces, significantly reducing compilation and optimization overhead.

## 📂 Repository Contents

The project is structured to handle the entire machine learning pipeline, from dataset generation to model training and evaluation.

| File/Folder | Description |
| :--- | :--- |
| **`modeling.py`** | **Model Definition.** Defines the LSTM architecture, including layers, activation functions, and output structure for regression (predicting time). |
| **`train_model.py`** | **Training Script.** Orchestrates the training loop. It loads data, initializes the model, optimizes weights, and saves checkpoints. |
| **`evaluate_model.py`** | **Evaluation.** Tests the trained model on a hold-out set, calculating metrics like MSE or MAPE to ensure generalization. |
| **`generate_dataset.py`**| **Data Gen.** Processes Halide program descriptions/schedules to extract measurable features for the LSTM input sequences. |
| `data_utils.py` | **Preprocessing.** Utility functions for loading, cleaning, normalizing, and sequencing raw features. |
| `train_utils.py` | **Helpers.** Training-specific utilities (custom learning rate schedules, checkpointing, metric logging). |
| `config.yaml` | **Configuration.** Central file for hyperparameters (batch size, LR), file paths, and environment settings. |
| `environment.yml` | **Dependencies.** Conda environment file to reproduce the exact software setup. |
| `data/` | Directory for storing raw feature logs and processed dataset binaries. |

## 🚀 Getting Started

### 1. Environment Setup
This project uses `conda` for dependency management. Create the environment using the provided file:

```bash
conda env create -f environment.yml
conda activate cost_model_env
```

### 2. Configuration
Open `config.yaml` to adjust hyperparameters or set the paths for your data:
```yaml
# Example config.yaml snippet
learning_rate: 0.001
batch_size: 64
epochs: 100
data_path: "./data/processed"
```

### 3. Workflow
The typical pipeline for running this project is:

1.  **Generate Data:** Extract features from your Halide schedules.
    ```bash
    python generate_dataset.py
    ```
2.  **Train:** Train the LSTM model on the generated data.
    ```bash
    python train_model.py
    ```
3.  **Evaluate:** Assess the model's accuracy on unseen schedules.
    ```bash
    python evaluate_model.py
    ```

## 🛠️ Tech Stack

*   **Language:** Python (100.0%)
*   **Architecture:** LSTM (Recurrent Neural Network)
*   **Input Data:** Halide Schedule Features
*   **Target Output:** Execution Time (Latency)
