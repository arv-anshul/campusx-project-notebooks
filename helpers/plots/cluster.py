"""
Plot cluster plot using `sklearn.cluster.Kmeans` class.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


def find_optimal_clusters(
    data: np.ndarray,
    *,
    cluster_range: range = ...,
) -> list[float]:
    """Find the optimal number of clusters using the Elbow Method."""
    if not isinstance(cluster_range, range):
        cluster_range = range(1, data.shape[1] + 1)

    wcss = []  # Within-Cluster-Sum-of-Squares
    for i in cluster_range:
        kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    return wcss


def plot_clusters(
    wcss: list[float],
    *,
    cluster_range: range = ...,
    title: str = ...,
) -> None:
    """Plot the Elbow Method results."""
    if not isinstance(cluster_range, range):
        cluster_range = range(1, len(wcss) + 1)

    plt.plot(cluster_range, wcss, marker="o", linestyle="--")

    plt.title(
        title
        if isinstance(title, str)
        else "Elbow Method for Optimal Number of Clusters"
    )
    plt.xlabel("Number of Clusters")
    plt.ylabel("WCSS")
    plt.show()


def fit_kmeans_model(data: np.ndarray, best_n_clusters: int) -> KMeans:
    """Fit the KMeans model with the best number of clusters."""
    kmeans = KMeans(
        n_clusters=best_n_clusters, init="k-means++", n_init=10, random_state=42
    )
    kmeans.fit(data)
    return kmeans
