"""
ELI5 (Explain Like I'm 5) implementation for model explanations.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import eli5
from eli5.sklearn import explain_prediction, explain_weights
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator

from src.models.base_model import BaseXAIModel
from src.utils.visualization import plot_feature_importance


class ELI5Explainer(BaseXAIModel):
    """ELI5 explainer implementation."""

    def __init__(
        self,
        model: BaseEstimator,
        top_n: int = 10,
    ):
        """
        Initialize ELI5 explainer.

        Args:
            model: The underlying machine learning model
            top_n: Number of top features to include in explanation
        """
        super().__init__(model)
        self.top_n = top_n

    def fit(self) -> None:
        """Fit the model."""
        # Load and train the model
        self.load_data()
        self.train()

    def explain_instance(
        self, instance: np.ndarray, top_n: Optional[int] = None
    ) -> Tuple[Dict[str, float], Figure]:
        """
        Explain a single instance prediction.

        Args:
            instance: Instance to explain
            top_n: Number of top features to include in explanation

        Returns:
            Tuple of (explanation dictionary, feature importance plot)
        """
        if top_n is None:
            top_n = self.top_n

        # Get ELI5 explanation
        explanation = explain_prediction(
            self.model,
            instance,
            feature_names=self.feature_names,
            top=top_n,
        )
        
        # Extract feature importance from explanation
        feature_importance = {}
        for feature in explanation.targets[0].feature_weights.pos:
            feature_importance[feature.name] = feature.weight
            
        for feature in explanation.targets[0].feature_weights.neg:
            feature_importance[feature.name] = feature.weight
            
        # Sort by absolute value
        feature_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        )
        
        # Create feature importance plot
        features = list(feature_importance.keys())
        importance_scores = np.array(list(feature_importance.values()))
        fig = plot_feature_importance(
            importance_scores,
            features,
            title="ELI5 Feature Importance",
            top_n=top_n,
        )
        
        return feature_importance, fig

    def explain_batch(
        self, instances: np.ndarray, top_n: Optional[int] = None
    ) -> List[Tuple[Dict[str, float], Figure]]:
        """
        Explain multiple instances.

        Args:
            instances: Array of instances to explain
            top_n: Number of top features to include in explanation

        Returns:
            List of (explanation dictionary, feature importance plot) tuples
        """
        explanations = []
        for instance in instances:
            explanation = self.explain_instance(instance, top_n)
            explanations.append(explanation)
        return explanations
    
    def explain_global(self) -> Tuple[Dict[str, float], Figure]:
        """
        Generate global feature importance explanation.

        Returns:
            Tuple of (explanation dictionary, feature importance plot)
        """
        # Get ELI5 global explanation
        explanation = explain_weights(
            self.model,
            feature_names=self.feature_names,
            top=self.top_n,
        )
        
        # Extract feature importance from explanation
        feature_importance = {}
        for feature in explanation.targets[0].feature_weights.pos:
            feature_importance[feature.name] = feature.weight
            
        for feature in explanation.targets[0].feature_weights.neg:
            feature_importance[feature.name] = feature.weight
            
        # Sort by absolute value
        feature_importance = dict(
            sorted(
                feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
        )
        
        # Create feature importance plot
        features = list(feature_importance.keys())
        importance_scores = np.array(list(feature_importance.values()))
        fig = plot_feature_importance(
            importance_scores,
            features,
            title="ELI5 Global Feature Importance",
            top_n=self.top_n,
        )
        
        return feature_importance, fig
    
    def format_explanation(self, explanation_dict: Dict[str, float]) -> str:
        """
        Format explanation dictionary as a human-readable string.

        Args:
            explanation_dict: Dictionary of feature importance scores

        Returns:
            Formatted explanation string
        """
        result = "ELI5 Explanation:\n\n"
        
        # Add positive contributions
        pos_features = {k: v for k, v in explanation_dict.items() if v > 0}
        if pos_features:
            result += "Features that contribute positively to the prediction:\n"
            for feature, importance in pos_features.items():
                result += f"  - {feature}: +{importance:.4f}\n"
            result += "\n"
            
        # Add negative contributions
        neg_features = {k: v for k, v in explanation_dict.items() if v < 0}
        if neg_features:
            result += "Features that contribute negatively to the prediction:\n"
            for feature, importance in neg_features.items():
                result += f"  - {feature}: {importance:.4f}\n"
            result += "\n"
            
        return result 