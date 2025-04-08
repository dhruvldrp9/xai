#!/usr/bin/env python
"""
Alibi Demo Script

This script demonstrates how to use Alibi to explain predictions of a machine learning model
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

from src.xai.alibi.alibi_explainer import AlibiExplainer


def main():
    """Main function to demonstrate Alibi explanations."""
    print("=" * 80)
    print("Alibi Demo")
    print("=" * 80)
    
    # Initialize the model
    print("\n1. Initializing and training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Try both explainer types
    for explainer_type in ["anchor", "counterfactual"]:
        print(f"\n\n{'=' * 40}")
        print(f"Using {explainer_type} explainer")
        print(f"{'=' * 40}")
        
        explainer = AlibiExplainer(model, explainer_type=explainer_type)
        
        # Fit the model
        explainer.fit()
        
        # Make predictions on test set
        y_pred = explainer.predict(explainer.X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(explainer.y_test, y_pred))
        
        # Explain individual predictions
        print(f"\n2. Explaining individual predictions using {explainer_type}...")
        num_instances = 2  # Limit to 2 instances for counterfactual due to computation time
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
            fig.savefig(f"alibi_{explainer_type}_instance_{i+1}.png")
            print(f"Figure saved as alibi_{explainer_type}_instance_{i+1}.png")
    
    # Print Alibi limitations and best practices
    print("\n3. Alibi Limitations and Best Practices:")
    print("\nLimitations:")
    print("1. Anchor explanations may be computationally expensive for large datasets")
    print("2. Counterfactual explanations may not always find a valid counterfactual")
    print("3. May require careful tuning of hyperparameters for optimal results")
    
    print("\nBest Practices:")
    print("1. Use Anchor explanations for understanding decision rules")
    print("2. Use Counterfactual explanations for understanding what changes would alter the prediction")
    print("3. Consider using both types of explanations for a more complete understanding")
    print("4. Adjust hyperparameters based on the specific use case and model")


if __name__ == "__main__":
    main() 