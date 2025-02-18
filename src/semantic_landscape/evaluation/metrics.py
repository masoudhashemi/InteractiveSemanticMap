from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


class ClusterEvaluator:
    """Evaluator for clustering quality"""

    def __init__(self, embeddings: np.ndarray):
        """
        Initialize evaluator with document embeddings

        Args:
            embeddings: Document embeddings matrix
        """
        self.embeddings = embeddings

    def calculate_cluster_coherence(self, clusters: Dict[Tuple[int, int], List[int]]) -> Dict[Tuple[int, int], float]:
        """
        Calculate internal coherence for each cluster

        Args:
            clusters: Dictionary mapping positions to lists of document indices

        Returns:
            Dictionary mapping positions to coherence scores
        """
        coherence_scores = {}

        for pos, doc_indices in clusters.items():
            if len(doc_indices) < 2:
                coherence_scores[pos] = 1.0  # Perfect coherence for single-document clusters
                continue

            # Get embeddings for this cluster
            cluster_embeddings = self.embeddings[doc_indices]

            # Calculate pairwise similarities
            similarities = cosine_similarity(cluster_embeddings)

            # Average similarity (excluding self-similarity)
            np.fill_diagonal(similarities, 0)
            coherence = similarities.sum() / (len(doc_indices) * (len(doc_indices) - 1))

            coherence_scores[pos] = coherence

        return coherence_scores

    def calculate_cluster_separation(self, clusters: Dict[Tuple[int, int], List[int]]) -> float:
        """
        Calculate separation between clusters using Calinski-Harabasz score

        Args:
            clusters: Dictionary mapping positions to lists of document indices

        Returns:
            Cluster separation score
        """
        if len(clusters) < 2:
            return 0.0

        # Create cluster labels array
        labels = np.zeros(len(self.embeddings), dtype=int)
        for i, doc_indices in enumerate(clusters.values()):
            for idx in doc_indices:
                labels[idx] = i

        return calinski_harabasz_score(self.embeddings, labels)

    def calculate_silhouette(self, clusters: Dict[Tuple[int, int], List[int]]) -> float:
        """
        Calculate silhouette score for clustering

        Args:
            clusters: Dictionary mapping positions to lists of document indices

        Returns:
            Silhouette score
        """
        if len(clusters) < 2:
            return 0.0

        # Create cluster labels array
        labels = np.zeros(len(self.embeddings), dtype=int)
        for i, doc_indices in enumerate(clusters.values()):
            for idx in doc_indices:
                labels[idx] = i

        return silhouette_score(self.embeddings, labels, metric="cosine")
