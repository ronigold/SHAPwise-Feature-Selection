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
from sklearn.metrics import accuracy_score
from your_model import YourModel
import pandas as pd

# Load your dataset
X, y = pd.read_csv("your_dataset.csv").drop("target", axis=1), pd.read_csv("your_dataset.csv")["target"]

# Initialize your model
model = YourModel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize SHAPwise Feature Selector
sfs_model = SHAPwiseFeatureSelector(model, accuracy_score)

# Perform feature selection

# Train your model using selected features
sfs_model.fit(X_train, y_train, X_test, y_test)

drop_features = sfs_model.sfs_model
model = sfs_model.base_estimator
```

### Advanced Configuration

Refer to the documentation for advanced configuration options, including setting thresholds for SHAP value correlation and customizing the iterative selection process.

## Documentation

For detailed documentation, including API references and advanced usage examples, please visit [SFS Documentation](https://github.com/yourusername/sfs-shapwise-feature-selection/docs).

## Contributing

We welcome contributions to the SFS project! Please refer to our contribution guidelines for more information on how to report issues, submit pull requests, and more.

## License

SFS is released under the MIT License. See the LICENSE file for more details.

## Acknowledgments

This project was inspired by the work on SHapley Additive exPlanations (SHAP) and aims to bring the power of SHAP values to the realm of feature selection for machine learning models.

## Citation

If you use SFS in your research, please cite our paper:

```plaintext
@article{yourname2024shapwise,
  title={SHAPwise Feature Selection (SFS): Enhancing Predictive Models through Correlation-Based Feature Analysis},
  author={Your Name and Collaborators},
  journal={Journal of Machine Learning Research},
  volume={xx},
  number={xx},
  pages={xx-xx},
  year={2024}
}
```

For further information and support, please open an issue in the GitHub repository.

