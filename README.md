# XAI Toolkit

A comprehensive toolkit for implementing and comparing various Explainable AI (XAI) frameworks in Python. This project demonstrates how to make machine learning models more interpretable and transparent using multiple popular XAI techniques.

## Features

- Implementation of 4 popular XAI frameworks:
  - LIME (Local Interpretable Model-agnostic Explanations)
  - SHAP (SHapley Additive exPlanations)
  - ELI5 (Explain Like I'm 5)
  - Alibi
- Consistent implementation across frameworks using the Breast Cancer Wisconsin dataset
- Detailed visualizations and interpretations
- Comprehensive documentation and best practices

## Project Structure

```
xai_toolkit/
├── data/                  # Dataset and data processing scripts
├── notebooks/            # Jupyter notebooks for each XAI framework
├── src/                  # Source code
│   ├── models/          # Model implementations
│   ├── xai/             # XAI framework implementations
│   │   ├── lime/       # LIME implementation
│   │   ├── shap/       # SHAP implementation
│   │   ├── eli5/       # ELI5 implementation
│   │   └── alibi/      # Alibi implementation
│   └── utils/           # Utility functions
├── tests/               # Unit tests
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/xai_toolkit.git
cd xai_toolkit
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Main Script

The main script demonstrates all four XAI frameworks:

```bash
# Run all demos
python src/main.py

# Run specific framework demo
python src/main.py --framework lime
python src/main.py --framework shap
python src/main.py --framework eli5
python src/main.py --framework alibi

# Show framework comparison only
python src/main.py --framework compare
```

### Using Individual Framework Scripts

Each framework has its own demo script:

```bash
# LIME demo
python src/xai/lime/lime_demo.py

# SHAP demo
python src/xai/shap/shap_demo.py

# ELI5 demo
python src/xai/eli5/eli5_demo.py

# Alibi demo
python src/xai/alibi/alibi_demo.py
```

### Using the XAI Framework Classes

You can use the XAI framework classes in your own code:

```python
from src.xai.lime.lime_explainer import LIMEExplainer
from src.xai.shap.shap_explainer import SHAPExplainer
from src.xai.eli5.eli5_explainer import ELI5Explainer
from src.xai.alibi.alibi_explainer import AlibiExplainer
from sklearn.ensemble import RandomForestClassifier

# Initialize a model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Create an explainer
explainer = LIMEExplainer(model)  # or SHAPExplainer, ELI5Explainer, AlibiExplainer

# Fit the explainer
explainer.fit()

# Explain a single instance
explanation_dict, fig = explainer.explain_instance(instance)

# Explain multiple instances
explanations = explainer.explain_batch(instances)

# Get global feature importance (if supported)
global_importance, global_fig = explainer.explain_global()
```

## Framework Comparison

Each XAI framework has its strengths and limitations:

- **LIME**: Best for local explanations of individual predictions
- **SHAP**: Excellent for both local and global feature importance
- **ELI5**: Great for simple, intuitive explanations
- **Alibi**: Specialized in detecting model drift and providing counterfactual explanations

See the individual notebooks for detailed comparisons and use cases.

## Examples

### LIME Explanation

LIME provides local explanations for individual predictions:

```
Instance 1:
True label: 1
Predicted label: 1

Feature Importance:
mean radius: 0.1234
mean texture: -0.0567
mean perimeter: 0.0890
...
```

### SHAP Explanation

SHAP provides both local and global feature importance:

```
Global Feature Importance:
mean radius: 0.2345
mean texture: 0.1234
mean perimeter: 0.3456
...
```

### ELI5 Explanation

ELI5 provides human-readable explanations:

```
ELI5 Explanation:

Features that contribute positively to the prediction:
  - mean radius: +0.1234
  - mean perimeter: +0.0890

Features that contribute negatively to the prediction:
  - mean texture: -0.0567
  - mean smoothness: -0.0345
```

### Alibi Explanation

Alibi provides anchor and counterfactual explanations:

```
Alibi Explanation:

Anchor: IF mean radius > 15.2 AND mean perimeter > 95.3 THEN malignant
Precision: 0.9876
Coverage: 0.3456

Feature Importance:
  - mean radius: 0.2345
  - mean perimeter: 0.1234
  - mean texture: 0.0567
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
