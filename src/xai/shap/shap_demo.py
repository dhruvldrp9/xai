#!/usr/bin/env python
"""
SHAP (SHapley Additive exPlanations) Demo Script

This script demonstrates how to use SHAP to explain predictions of a machine learning model
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

from src.xai.shap.shap_explainer import SHAPExplainer


def main():
    """Main function to demonstrate SHAP explanations."""
    print("=" * 80)
    print("SHAP (SHapley Additive exPlanations) Demo")
    print("=" * 80)
    
    # Initialize the model
    print("\n1. Initializing and training the model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    explainer = SHAPExplainer(model, explainer_type="tree")
    
    # Fit the model and initialize SHAP explainer
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
        fig.savefig(f"shap_explanation_instance_{i+1}.png")
        print(f"Figure saved as shap_explanation_instance_{i+1}.png")
    
    # Generate global feature importance
    print("\n3. Generating global feature importance...")
    mean_shap_values, global_fig = explainer.explain_global()
    
    # Display global feature importance
    print("\nGlobal Feature Importance:")
    for i, (feature, importance) in enumerate(zip(explainer.feature_names, mean_shap_values)):
        print(f"{feature}: {importance:.4f}")
    
    # Save the global feature importance figure
    global_fig.savefig("shap_global_feature_importance.png")
    print("Global feature importance saved as shap_global_feature_importance.png")
    
    # Create and save SHAP summary plot
    print("\n4. Creating SHAP summary plot...")
    summary_fig = explainer.plot_summary()
    summary_fig.savefig("shap_summary_plot.png")
    print("SHAP summary plot saved as shap_summary_plot.png")
    
    # Create and save SHAP dependence plot for the most important feature
    print("\n5. Creating SHAP dependence plot...")
    most_important_feature_idx = np.argmax(mean_shap_values)
    dependence_fig = explainer.plot_dependence(most_important_feature_idx)
    dependence_fig.savefig("shap_dependence_plot.png")
    print(f"SHAP dependence plot for {explainer.feature_names[most_important_feature_idx]} saved as shap_dependence_plot.png")
    
    # Print SHAP limitations and best practices
    print("\n6. SHAP Limitations and Best Practices:")
    print("\nLimitations:")
    print("1. Computationally expensive for large datasets and complex models")
    print("2. May be difficult to interpret for high-dimensional data")
    print("3. Assumes feature independence, which may not hold in all cases")
    
    print("\nBest Practices:")
    print("1. Use appropriate explainer type based on the model (tree, kernel, or deep)")
    print("2. Consider using background samples for kernel explainer to reduce computation time")
    print("3. Use summary plots to understand global feature importance")
    print("4. Use dependence plots to understand feature interactions")


if __name__ == "__main__":
    main() 