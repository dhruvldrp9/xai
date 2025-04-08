"""
LIME implementation for model explanations.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from lime import lime_tabular
from sklearn.base import BaseEstimator

from src.models.base_model import BaseXAIModel
from src.utils.visualization import plot_feature_importance


class LIMEExplainer(BaseXAIModel):
    """LIME explainer implementation."""

    def __init__(
        self,
        model: BaseEstimator,
        num_features: int = 10,
        num_samples: int = 5000,
    ):
        """
        Initialize LIME explainer.

        Args:
            model: The underlying machine learning model
            num_features: Number of features to use in explanation
            num_samples: Number of samples to generate for LIME
        """
        super().__init__(model)
        self.num_features = num_features
        self.num_samples = num_samples
        self.explainer = None

    def fit(self) -> None:
        """Fit the model and initialize LIME explainer."""
        # Load and train the model
        self.load_data()
        self.train()

        # Initialize LIME explainer
        self.explainer = lime_tabular.LimeTabularExplainer(
            self.X_train,
            feature_names=self.feature_names,
            class_names=["Benign", "Malignant"],
            mode="classification",
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

        if num_features is None:
            num_features = self.num_features

        # Get LIME explanation
        exp = self.explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features,
        )

        # Convert explanation to dictionary
        explanation_dict = dict(exp.as_list())

        # Create feature importance plot
        features = list(explanation_dict.keys())
        importance_scores = np.array(list(explanation_dict.values()))
        fig = plot_feature_importance(
            importance_scores,
            features,
            title="LIME Feature Importance",
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