"""
Utility functions for visualizing XAI explanations.
"""
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


def plot_feature_importance(
    importance_scores: np.ndarray,
    feature_names: List[str],
    title: str = "Feature Importance",
    top_n: Optional[int] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> Figure:
    """
    Plot feature importance scores.

    Args:
        importance_scores: Array of importance scores
        feature_names: List of feature names
        title: Plot title
        top_n: Number of top features to display
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    if top_n is not None:
        idx = np.argsort(importance_scores)[-top_n:]
        importance_scores = importance_scores[idx]
        feature_names = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=importance_scores, y=feature_names, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return fig


def plot_explanation_heatmap(
    explanation_matrix: np.ndarray,
    feature_names: List[str],
    title: str = "Explanation Heatmap",
    figsize: Tuple[int, int] = (12, 8),
) -> Figure:
    """
    Plot explanation heatmap.

    Args:
        explanation_matrix: Matrix of explanation values
        feature_names: List of feature names
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        explanation_matrix,
        xticklabels=feature_names,
        yticklabels=False,
        cmap="RdBu",
        center=0,
        ax=ax,
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


def plot_prediction_distribution(
    predictions: np.ndarray,
    title: str = "Prediction Distribution",
    figsize: Tuple[int, int] = (8, 6),
) -> Figure:
    """
    Plot distribution of model predictions.

    Args:
        predictions: Array of model predictions
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(predictions, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Prediction Value")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig 