#!/usr/bin/env python
"""
LIME (Local Interpretable Model-agnostic Explanations) Demo Script

This script demonstrates how to use LIME to explain predictions of a machine learning model
on the Breast Cancer Wisconsin dataset.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.xai.lime.lime_explainer import LIMEExplainer


def main():
    """Main function to demonstrate LIME explanations."""
    print("=" * 80)
    print("LIME (Local Interpretable Model-agnostic Explanations) Demo")
    print("=" * 80)
    
    # Initialize the model
    print("\n1. Initializing and training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    explainer = LIMEExplainer(model)
    
    # Fit the model and initialize LIME explainer
    explainer.fit()
    
    # Make predictions on test set
    y_pred = explainer.predict(explainer.X_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(explainer.y_test, y_pred))
    
    # Explain individual predictions
    print("\n2. Explaining individual predictions...")
    num_instances = 3
    instances = explainer.X_test[:num_instances]
    
    # Get explanations for each instance
    explanations = explainer.explain_batch(instances)
    
    # Display explanations
    for i, (explanation_dict, fig) in enumerate(explanations):
        print(f"\nInstance {i+1}:")
        print(f"True label: {explainer.y_test[i]}")
        print(f"Predicted label: {explainer.predict(instances[i:i+1])[0]}")
        print("\nFeature Importance:")
        for feature, importance in explanation_dict.items():
            print(f"{feature}: {importance:.4f}")
        
        # Save the figure
        fig.savefig(f"lime_explanation_instance_{i+1}.png")
        print(f"Figure saved as lime_explanation_instance_{i+1}.png")
    
    # Analyze feature importance patterns
    print("\n3. Analyzing feature importance patterns...")
    num_instances = 10
    instances = explainer.X_test[:num_instances]
    explanations = explainer.explain_batch(instances)
    
    # Collect feature importance scores
    feature_importance_matrix = np.zeros((num_instances, len(explainer.feature_names)))
    for i, (explanation_dict, _) in enumerate(explanations):
        for feature, importance in explanation_dict.items():
            feature_idx = np.where(explainer.feature_names == feature)[0][0]
            feature_importance_matrix[i, feature_idx] = importance
    
    # Plot heatmap of feature importance
    plt.figure(figsize=(15, 8))
    plt.imshow(feature_importance_matrix, aspect='auto', cmap='RdBu', interpolation='nearest')
    plt.colorbar()
    plt.title('Feature Importance Patterns Across Instances')
    plt.xlabel('Features')
    plt.ylabel('Instances')
    plt.xticks(range(len(explainer.feature_names)), explainer.feature_names, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("lime_feature_importance_patterns.png")
    print("Feature importance patterns saved as lime_feature_importance_patterns.png")
    
    # Print LIME limitations and best practices
    print("\n4. LIME Limitations and Best Practices:")
    print("\nLimitations:")
    print("1. Local approximations may not capture global model behavior")
    print("2. Sensitive to the choice of kernel width and number of samples")
    print("3. May not work well with highly non-linear models")
    
    print("\nBest Practices:")
    print("1. Use sufficient number of samples for stable explanations")
    print("2. Consider the scale of features when interpreting importance scores")
    print("3. Compare explanations across multiple instances to identify patterns")
    print("4. Use in conjunction with other XAI methods for a more complete understanding")


if __name__ == "__main__":
    main() 