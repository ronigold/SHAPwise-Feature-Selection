# SHAPwise Feature Selection (SFS)

## Overview

SHAPwise Feature Selection (SFS) is a Python library designed to enhance predictive model performance by leveraging SHapley Additive exPlanations (SHAP) values for feature selection. This innovative method systematically evaluates the impact of features on model predictions, identifying and excluding less impactful ones to refine the feature set. The result is not only an improvement in model accuracy but also in interpretability and efficiency.

## Features

- **SHAP Value Analysis**: Utilizes SHAP values to quantify the contribution of each feature towards the model's predictions.
- **Iterative Feature Selection**: Employs an iterative process to remove features with low correlation between their SHAP values and their impact on the model outcome.
- **Model Performance Enhancement**: Aims to improve model performance by retaining only the most impactful features.
- **Compatibility**: Designed for use with PyTorch and FastAI, facilitating easy integration into existing machine learning workflows.
- **Interpretability Improvement**: Enhances model interpretability by focusing on features that significantly affect model predictions.

## Installation

```bash
pip install sfs-shapwise-feature-selection
```

## Usage

### Basic Example

```python
from sfs import SHAPwiseFeatureSelector
from your_model import YourModel
import pandas as pd

# Load your dataset
X, y = pd.read_csv("your_dataset.csv").drop("target", axis=1), pd.read_csv("your_dataset.csv")["target"]

# Initialize your model
model = YourModel()

# Initialize SHAPwise Feature Selector
sfs = SHAPwiseFeatureSelector(model, X, y)

# Perform feature selection
selected_features = sfs.select_features()

# Train your model using selected features
model.fit(X[selected_features], y)

