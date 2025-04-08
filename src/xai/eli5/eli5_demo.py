#!/usr/bin/env python
"""
ELI5 (Explain Like I'm 5) Demo Script

This script demonstrates how to use ELI5 to explain predictions of a machine learning model
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

from src.xai.eli5.eli5_explainer import ELI5Explainer


def main():
    """Main function to demonstrate ELI5 explanations."""
    print("=" * 80)
    print("ELI5 (Explain Like I'm 5) Demo")
    print("=" * 80)
    
    # Initialize the model
    print("\n1. Initializing and training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    explainer = ELI5Explainer(model)
    
    # Fit the model
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
        
        # Format and print the explanation
        formatted_explanation = explainer.format_explanation(explanation_dict)
        print(formatted_explanation)
        
        # Save the figure
        fig.savefig(f"eli5_explanation_instance_{i+1}.png")
        print(f"Figure saved as eli5_explanation_instance_{i+1}.png")
    
    # Generate global feature importance
    print("\n3. Generating global feature importance...")
    global_importance, global_fig = explainer.explain_global()
    
    # Format and print the global explanation
    formatted_global = explainer.format_explanation(global_importance)
    print(formatted_global)
    
    # Save the global feature importance figure
    global_fig.savefig("eli5_global_feature_importance.png")
    print("Global feature importance saved as eli5_global_feature_importance.png")
    
    # Print ELI5 limitations and best practices
    print("\n4. ELI5 Limitations and Best Practices:")
    print("\nLimitations:")
    print("1. Limited to scikit-learn compatible models")
    print("2. May not provide as detailed explanations as LIME or SHAP")
    print("3. Less flexible for custom model types")
    
    print("\nBest Practices:")
    print("1. Use for simple, intuitive explanations of model decisions")
    print("2. Leverage the human-readable format for non-technical stakeholders")
    print("3. Combine with other XAI methods for more comprehensive explanations")
    print("4. Use for quick model debugging and understanding")


if __name__ == "__main__":
    main() 