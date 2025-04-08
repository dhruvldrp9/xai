"""
Base model class for XAI implementations.
"""
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class BaseXAIModel:
    """Base class for XAI model implementations."""

    def __init__(self, model: BaseEstimator):
        """
        Initialize the base XAI model.

        Args:
            model: The underlying machine learning model
        """
        self.model = model
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load and preprocess the Breast Cancer Wisconsin dataset.

        Returns:
            Tuple containing X_train, X_test, y_train, y_test
        """
        # Load the breast cancer dataset
        data = load_breast_cancer()
        X, y = data.data, data.target
        self.feature_names = data.feature_names

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale the features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        return X_train, X_test, y_train, y_test

    def train(self) -> None:
        """Train the model on the loaded data."""
        if self.X_train is None:
            self.load_data()
        self.model.fit(self.X_train, self.y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Args:
            X: Input features

        Returns:
            Model predictions
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions using the trained model.

        Args:
            X: Input features

        Returns:
            Probability predictions
        """
        return self.model.predict_proba(X)

    def get_feature_names(self) -> np.ndarray:
        """
        Get the feature names.

        Returns:
            Array of feature names
        """
        return self.feature_names 