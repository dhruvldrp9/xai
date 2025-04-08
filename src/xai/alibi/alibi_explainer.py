"""
Alibi implementation for model explanations.
"""
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from alibi.explainers import AnchorTabular, CounterFactual
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator

from src.models.base_model import BaseXAIModel
from src.utils.visualization import plot_feature_importance


class AlibiExplainer(BaseXAIModel):
    """Alibi explainer implementation."""

    def __init__(
        self,
        model: BaseEstimator,
        explainer_type: str = "anchor",
        num_features: int = 10,
    ):
        """
        Initialize Alibi explainer.

        Args:
            model: The underlying machine learning model
            explainer_type: Type of Alibi explainer to use ('anchor' or 'counterfactual')
            num_features: Number of features to include in explanation
        """
        super().__init__(model)
        self.explainer_type = explainer_type
        self.num_features = num_features
        self.explainer = None
        self.categorical_names = {}  # No categorical features in our dataset

    def fit(self) -> None:
        """Fit the model and initialize Alibi explainer."""
        # Load and train the model
        self.load_data()
        self.train()

        # Initialize Alibi explainer based on type
        if self.explainer_type == "anchor":
            self.explainer = AnchorTabular(
                self.model.predict_proba,
                self.feature_names,
                categorical_names=self.categorical_names,
            )
            self.explainer.fit(self.X_train)
        elif self.explainer_type == "counterfactual":
            self.explainer = CounterFactual(
                self.model.predict_proba,
                shape=(1, self.X_train.shape[1]),
                target_proba=0.5,
                target_class=None,
                lam_init=1e-1,
                max_iter=1000,
                early_stop=50,
                lam=1e-1,
                max_lam_steps=10,
                learning_rate_init=0.1,
                feature_range=(self.X_train.min(axis=0), self.X_train.max(axis=0)),
            )
        else:
            raise ValueError(
                f"Unsupported explainer type: {self.explainer_type}. "
                "Supported types are 'anchor' and 'counterfactual'."
            )

    def explain_instance(
        self, instance: np.ndarray, num_features: Optional[int] = None
    ) -> Tuple[Dict[str, Any], Figure]:
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

        if num_features is None:
            num_features = self.num_features

        # Get explanation based on explainer type
        if self.explainer_type == "anchor":
            explanation = self.explainer.explain(instance)
            
            # Extract feature importance from explanation
            feature_importance = {}
            for feature, importance in explanation.data["feature_importance"].items():
                feature_importance[feature] = importance
                
            # Sort by absolute value
            feature_importance = dict(
                sorted(
                    feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
            )
            
            # Limit to top features
            feature_importance = dict(list(feature_importance.items())[:num_features])
            
            # Create feature importance plot
            features = list(feature_importance.keys())
            importance_scores = np.array(list(feature_importance.values()))
            fig = plot_feature_importance(
                importance_scores,
                features,
                title="Alibi Anchor Feature Importance",
                top_n=num_features,
            )
            
            # Add anchor information to explanation
            explanation_dict = {
                "feature_importance": feature_importance,
                "anchor": explanation.data["anchor"],
                "precision": explanation.data["precision"],
                "coverage": explanation.data["coverage"],
            }
            
        elif self.explainer_type == "counterfactual":
            explanation = self.explainer.explain(instance)
            
            # Extract counterfactual information
            cf = explanation.data["cf"]
            original = explanation.data["original"]
            
            # Calculate feature importance as absolute difference
            feature_importance = {}
            for i, feature in enumerate(self.feature_names):
                feature_importance[feature] = abs(cf[0, i] - original[0, i])
                
            # Sort by value
            feature_importance = dict(
                sorted(
                    feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            )
            
            # Limit to top features
            feature_importance = dict(list(feature_importance.items())[:num_features])
            
            # Create feature importance plot
            features = list(feature_importance.keys())
            importance_scores = np.array(list(feature_importance.values()))
            fig = plot_feature_importance(
                importance_scores,
                features,
                title="Alibi Counterfactual Feature Changes",
                top_n=num_features,
            )
            
            # Add counterfactual information to explanation
            explanation_dict = {
                "feature_importance": feature_importance,
                "counterfactual": cf[0],
                "original": original[0],
                "prediction": explanation.data["prediction"],
                "counterfactual_prediction": explanation.data["cf_prediction"],
            }
            
        return explanation_dict, fig

    def explain_batch(
        self, instances: np.ndarray, num_features: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], Figure]]:
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
    
    def format_explanation(self, explanation_dict: Dict[str, Any]) -> str:
        """
        Format explanation dictionary as a human-readable string.

        Args:
            explanation_dict: Dictionary of explanation data

        Returns:
            Formatted explanation string
        """
        result = "Alibi Explanation:\n\n"
        
        if self.explainer_type == "anchor":
            result += f"Anchor: {explanation_dict['anchor']}\n"
            result += f"Precision: {explanation_dict['precision']:.4f}\n"
            result += f"Coverage: {explanation_dict['coverage']:.4f}\n\n"
            
            result += "Feature Importance:\n"
            for feature, importance in explanation_dict["feature_importance"].items():
                result += f"  - {feature}: {importance:.4f}\n"
                
        elif self.explainer_type == "counterfactual":
            result += f"Original prediction: {explanation_dict['prediction']}\n"
            result += f"Counterfactual prediction: {explanation_dict['counterfactual_prediction']}\n\n"
            
            result += "Feature Changes (Original → Counterfactual):\n"
            for feature, importance in explanation_dict["feature_importance"].items():
                feature_idx = np.where(self.feature_names == feature)[0][0]
                original_val = explanation_dict["original"][feature_idx]
                cf_val = explanation_dict["counterfactual"][feature_idx]
                result += f"  - {feature}: {original_val:.4f} → {cf_val:.4f} (change: {importance:.4f})\n"
                
        return result 