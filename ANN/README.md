# ANN

This folder contains the artificial neural network workflow for the heart disease project. The main pipeline fetches the UCI Heart Disease dataset, cleans and encodes the features, trains a multilayer perceptron in PyTorch, evaluates the model, and saves the trained artifacts for inference in Streamlit.

## What the ANN does

The network is a binary classifier that predicts whether a patient is likely to have heart disease. It uses 13 clinical input features from the dataset and learns a nonlinear decision boundary through fully connected layers, ReLU activations, dropout regularization, and a sigmoid output layer.

The current training script follows these steps:

1. Load the heart disease dataset from `ucimlrepo`.
2. Inspect the raw data, missing values, and class balance.
3. Clean the data by replacing suspicious zero values and imputing missing values.
4. One-hot encode categorical features such as chest pain type, resting ECG, slope, and thal.
5. Split the data into training and test sets.
6. Standardize the features with `StandardScaler`.
7. Train a PyTorch multilayer perceptron with early stopping.
8. Evaluate the model with classification metrics, ROC-AUC, and confusion matrix plots.
9. Save the trained model and scaler for reuse in the app.

## Model architecture

The MLP currently uses this layout:

- Input layer: one neuron per processed feature
- Hidden layer 1: 128 units + ReLU + dropout
- Hidden layer 2: 64 units + ReLU + dropout
- Hidden layer 3: 32 units + ReLU
- Output layer: 1 unit + sigmoid

The sigmoid output produces a probability between 0 and 1.

## Files

- [Heart_Disease_Detection.py](Heart_Disease_Detection.py): training, evaluation, and model export script
- [app.py](app.py): Streamlit inference app that loads the saved model and scaler
- [models/](models): saved `.pth` model weights and scaler files
- [plots/](plots): generated training and evaluation charts

## How to run

From the repository root:

```powershell
python ANN/Heart_Disease_Detection.py
streamlit run ANN/app.py
```

If you run the training script from a different working directory, make sure the paths for `ANN/models` and `ANN/plots` still resolve correctly.

## Feature notes

The dataset features include:

- `age`: continuous
- `sex`: binary
- `cp`: categorical chest pain type
- `trestbps`: resting blood pressure
- `chol`: cholesterol
- `fbs`: fasting blood sugar indicator
- `restecg`: resting ECG category
- `thalach`: maximum heart rate achieved
- `exang`: exercise-induced angina
- `oldpeak`: ST depression
- `slope`: slope of the peak exercise ST segment
- `ca`: number of major vessels colored by fluoroscopy
- `thal`: thalassemia category

## Notes

The project is intended as a learning and demo workflow, not a clinical diagnostic tool. The Streamlit app is for interactive inference only and should not be used for medical decisions.
