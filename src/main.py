#!/usr/bin/env python
"""
XAI Toolkit - Main Script

This script demonstrates the usage of all four XAI frameworks:
1. LIME (Local Interpretable Model-agnostic Explanations)
2. SHAP (SHapley Additive exPlanations)
3. ELI5 (Explain Like I'm 5)
4. Alibi

Each framework is used to explain predictions of a machine learning model
on the Breast Cancer Wisconsin dataset.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.xai.lime.lime_explainer import LIMEExplainer
from src.xai.shap.shap_explainer import SHAPExplainer
from src.xai.eli5.eli5_explainer import ELI5Explainer
from src.xai.alibi.alibi_explainer import AlibiExplainer


def run_lime_demo():
    """Run the LIME demo."""
    print("\n" + "=" * 80)
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
    num_instances = 2
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


def run_shap_demo():
    """Run the SHAP demo."""
    print("\n" + "=" * 80)
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
    num_instances = 2
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


def run_eli5_demo():
    """Run the ELI5 demo."""
    print("\n" + "=" * 80)
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
    num_instances = 2
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


def run_alibi_demo():
    """Run the Alibi demo."""
    print("\n" + "=" * 80)
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
        num_instances = 1  # Limit to 1 instance for counterfactual due to computation time
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


def compare_frameworks():
    """Compare the four XAI frameworks."""
    print("\n" + "=" * 80)
    print("XAI Framework Comparison")
    print("=" * 80)
    
    print("\n1. LIME (Local Interpretable Model-agnostic Explanations)")
    print("   - Best for: Local explanations of individual predictions")
    print("   - Strengths: Model-agnostic, intuitive explanations")
    print("   - Limitations: Local approximations may not capture global behavior")
    
    print("\n2. SHAP (SHapley Additive exPlanations)")
    print("   - Best for: Both local and global feature importance")
    print("   - Strengths: Theoretically grounded, consistent explanations")
    print("   - Limitations: Computationally expensive for large datasets")
    
    print("\n3. ELI5 (Explain Like I'm 5)")
    print("   - Best for: Simple, intuitive explanations")
    print("   - Strengths: Human-readable explanations, easy to understand")
    print("   - Limitations: Limited to scikit-learn compatible models")
    
    print("\n4. Alibi")
    print("   - Best for: Decision rules and counterfactual explanations")
    print("   - Strengths: Provides both anchor and counterfactual explanations")
    print("   - Limitations: May be computationally expensive")
    
    print("\nFramework Selection Guidelines:")
    print("1. For quick, intuitive explanations: Use ELI5")
    print("2. For detailed local explanations: Use LIME")
    print("3. For comprehensive feature importance: Use SHAP")
    print("4. For decision rules and counterfactuals: Use Alibi")
    print("5. For the most complete understanding: Use a combination of frameworks")


def main():
    """Main function to demonstrate all XAI frameworks."""
    parser = argparse.ArgumentParser(description="XAI Toolkit Demo")
    parser.add_argument(
        "--framework",
        choices=["all", "lime", "shap", "eli5", "alibi", "compare"],
        default="all",
        help="XAI framework to demonstrate",
    )
    args = parser.parse_args()
    
    if args.framework == "all" or args.framework == "lime":
        run_lime_demo()
    
    if args.framework == "all" or args.framework == "shap":
        run_shap_demo()
    
    if args.framework == "all" or args.framework == "eli5":
        run_eli5_demo()
    
    if args.framework == "all" or args.framework == "alibi":
        run_alibi_demo()
    
    if args.framework == "all" or args.framework == "compare":
        compare_frameworks()


if __name__ == "__main__":
    main() 