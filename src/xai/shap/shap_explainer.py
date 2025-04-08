"""
SHAP (SHapley Additive exPlanations) implementation for model explanations.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import shap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator

from src.models.base_model import BaseXAIModel
from src.utils.visualization import plot_feature_importance, plot_explanation_heatmap


class SHAPExplainer(BaseXAIModel):
    """SHAP explainer implementation."""

    def __init__(
        self,
        model: BaseEstimator,
        background_samples: int = 100,
        explainer_type: str = "tree",
    ):
        """
        Initialize SHAP explainer.

        Args:
            model: The underlying machine learning model
            background_samples: Number of background samples for KernelExplainer
            explainer_type: Type of SHAP explainer to use ('tree', 'kernel', or 'deep')
        """
        super().__init__(model)
        self.background_samples = background_samples
        self.explainer_type = explainer_type
        self.explainer = None
        self.shap_values = None

    def fit(self) -> None:
        """Fit the model and initialize SHAP explainer."""
        # Load and train the model
        self.load_data()
        self.train()

        # Initialize SHAP explainer based on model type
        if self.explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.explainer_type == "kernel":
            # For kernel explainer, we need background data
            background_data = shap.sample(self.X_train, self.background_samples)
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, background_data
            )
        elif self.explainer_type == "deep":
            # For deep learning models
            self.explainer = shap.DeepExplainer(self.model, self.X_train)
        else:
            raise ValueError(
                f"Unsupported explainer type: {self.explainer_type}. "
                "Supported types are 'tree', 'kernel', and 'deep'."
            )

    def explain_instance(
        self, instance: np.ndarray, num_features: Optional[int] = None
    ) -> Tuple[Dict[str, float], Figure]:
        """
        Explain a single instance prediction.

        Args:
            instance: Instance to explain
            num_features: Number of features to include in explanation

        Returns:
            Tuple of (explanation dictionary, feature importance plot)
        """
        if self.explainer is None:
            self.fit()

        # Get SHAP values for the instance
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))
        
        # For binary classification, we're interested in the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
            
        # Convert to dictionary
        explanation_dict = {
            self.feature_names[i]: shap_values[0, i]
            for i in range(len(self.feature_names))
        }
        
        # Sort by absolute value
        explanation_dict = dict(
            sorted(
                explanation_dict.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        )
        
        # Limit to top features if specified
        if num_features is not None:
            explanation_dict = dict(list(explanation_dict.items())[:num_features])
        
        # Create feature importance plot
        features = list(explanation_dict.keys())
        importance_scores = np.array(list(explanation_dict.values()))
        fig = plot_feature_importance(
            importance_scores,
            features,
            title="SHAP Feature Importance",
            top_n=num_features,
        )
        
        return explanation_dict, fig

    def explain_batch(
        self, instances: np.ndarray, num_features: Optional[int] = None
    ) -> List[Tuple[Dict[str, float], Figure]]:
        """
        Explain multiple instances.

        Args:
            instances: Array of instances to explain
            num_features: Number of features to include in explanation

        Returns:
            List of (explanation dictionary, feature importance plot) tuples
        """
        explanations = []
        for instance in instances:
            explanation = self.explain_instance(instance, num_features)
            explanations.append(explanation)
        return explanations
    
    def explain_global(self) -> Tuple[np.ndarray, Figure]:
        """
        Generate global feature importance explanation.

        Returns:
            Tuple of (SHAP values array, feature importance plot)
        """
        if self.explainer is None:
            self.fit()
            
        # Get SHAP values for all training data
        shap_values = self.explainer.shap_values(self.X_train)
        
        # For binary classification, we're interested in the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
            
        # Calculate mean absolute SHAP values for each feature
        mean_shap_values = np.abs(shap_values).mean(axis=0)
        
        # Create feature importance plot
        fig = plot_feature_importance(
            mean_shap_values,
            self.feature_names,
            title="SHAP Global Feature Importance",
        )
        
        return mean_shap_values, fig
    
    def plot_summary(self) -> Figure:
        """
        Create a SHAP summary plot.

        Returns:
            Matplotlib figure
        """
        if self.explainer is None:
            self.fit()
            
        # Get SHAP values for all training data
        shap_values = self.explainer.shap_values(self.X_train)
        
        # For binary classification, we're interested in the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
            
        # Create summary plot
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values,
            self.X_train,
            feature_names=self.feature_names,
            show=False,
        )
        plt.tight_layout()
        
        return fig
    
    def plot_dependence(
        self, feature_idx: int, interaction_feature_idx: Optional[int] = None
    ) -> Figure:
        """
        Create a SHAP dependence plot.

        Args:
            feature_idx: Index of the feature to plot
            interaction_feature_idx: Index of the feature to use for coloring

        Returns:
            Matplotlib figure
        """
        if self.explainer is None:
            self.fit()
            
        # Get SHAP values for all training data
        shap_values = self.explainer.shap_values(self.X_train)
        
        # For binary classification, we're interested in the positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
            
        # Create dependence plot
        fig = plt.figure(figsize=(10, 8))
        if interaction_feature_idx is not None:
            shap.dependence_plot(
                feature_idx,
                shap_values,
                self.X_train,
                feature_names=self.feature_names,
                interaction_index=interaction_feature_idx,
                show=False,
            )
        else:
            shap.dependence_plot(
                feature_idx,
                shap_values,
                self.X_train,
                feature_names=self.feature_names,
                show=False,
            )
        plt.tight_layout()
        
        return fig 