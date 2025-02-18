from typing import List, Optional, Tuple

import hdbscan
import numpy as np
from minisom import MiniSom


class InteractiveSOM:
    """Self-Organizing Map with interactive learning capabilities"""

    def __init__(self, input_dim, grid_size=(10, 10), use_hdbscan=False, min_cluster_size=5):
        """Initialize the SOM with given input dimension and grid size."""
        self.input_dim = input_dim
        self.grid_size = grid_size if isinstance(grid_size, tuple) else (grid_size, grid_size)
        self.som = MiniSom(x=self.grid_size[0], y=self.grid_size[1], input_len=input_dim, random_seed=42)
        self.document_mappings = {}
        self.document_data = {}
        self.use_hdbscan = use_hdbscan
        self.min_cluster_size = min_cluster_size

    def initialize_weights(self, data):
        if not self.use_hdbscan:
            # Default random initialization - let MiniSom handle it
            return

        # Use HDBSCAN to get initial clusters
        clusterer = hdbscan.HDBSCAN(min_cluster_size=self.min_cluster_size)
        cluster_labels = clusterer.fit_predict(data)

        # Initialize weights based on cluster centroids
        unique_clusters = np.unique(cluster_labels)
        n_clusters = len(unique_clusters[unique_clusters != -1])  # Exclude noise points

        # Calculate cluster centroids
        centroids = []
        for i in range(n_clusters):
            mask = cluster_labels == i
            if np.any(mask):
                centroids.append(np.mean(data[mask], axis=0))

        # Fill remaining nodes with random initialization
        remaining_nodes = self.grid_size[0] * self.grid_size[1] - len(centroids)
        if remaining_nodes > 0:
            random_weights = np.random.random((remaining_nodes, self.input_dim))
            centroids.extend(random_weights)

        # Reshape to SOM grid
        self.som._weights = np.array(centroids[: self.grid_size[0] * self.grid_size[1]]).reshape(
            self.grid_size[0], self.grid_size[1], self.input_dim
        )

    def force_document_position(self, doc_id: str, embedding: np.ndarray, target_position: Tuple[int, int]):
        """
        Force a document to a specific position by adjusting the SOM weights and neighborhood
        """
        # Store the document data
        self.document_data[doc_id] = embedding

        # Define parameters with more aggressive values
        sigma = 2.0  # Smaller sigma for more focused influence
        learning_rate = 1.0  # Maximum learning rate
        threshold = 0.5  # Stricter threshold
        max_iterations = 100

        # Get current weights at target position
        target_weights = self.som._weights[target_position[0]][target_position[1]]

        # Calculate the direction we need to move the weights
        weight_direction = target_weights - embedding

        # Precompute the grid indices and distances
        x_indices, y_indices = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), indexing="ij")
        distances = np.sqrt((x_indices - target_position[0]) ** 2 + (y_indices - target_position[1]) ** 2)

        # Calculate the influence of the target position
        influence = np.exp(-(distances**2) / (2 * sigma**2))

        # Set target position to exactly match embedding
        self.som._weights[target_position[0]][target_position[1]] = embedding

        # Iteratively adjust surrounding weights
        for iteration in range(max_iterations):
            # Create a repulsive effect around the target position
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    if (i, j) != target_position:  # Skip the target position
                        dist = np.sqrt((i - target_position[0]) ** 2 + (j - target_position[1]) ** 2)
                        influence = np.exp(-(dist**2) / (2 * sigma**2))

                        # Move surrounding weights away from embedding
                        self.som._weights[i][j] += learning_rate * influence * weight_direction

            # Check current position
            current_position = self.som.winner(embedding)
            current_distance = np.sqrt(
                (current_position[0] - target_position[0]) ** 2 + (current_position[1] - target_position[1]) ** 2
            )

            if current_distance <= threshold:
                break

            # Increase learning rate if we're stuck
            if iteration > 0 and iteration % 10 == 0:
                learning_rate *= 1.2

        self.document_mappings[doc_id] = self.som.winner(embedding)

    def get_position_by_id(self, doc_id: str) -> Tuple[int, int]:
        """Get position for a document by its ID"""
        if doc_id not in self.document_mappings:
            # If mapping doesn't exist yet, create it
            if doc_id in self.document_data:
                embedding = self.document_data[doc_id]
                self.document_mappings[doc_id] = self.som.winner(embedding)
            else:
                raise KeyError(f"Document {doc_id} not found in document data")

        return self.document_mappings[doc_id]

    def train(self, embeddings: np.ndarray, epochs: int = 100, document_ids: Optional[List[str]] = None):
        """Train the SOM on document embeddings"""
        # Initialize weights using HDBSCAN if enabled
        if self.use_hdbscan:
            self.initialize_weights(embeddings)

        # Train the SOM
        self.som.train(embeddings, epochs, verbose=True)

        # Store document data and get initial mappings
        if document_ids:
            self.document_data = {doc_id: embedding for doc_id, embedding in zip(document_ids, embeddings)}

            # Calculate mappings based on trained SOM
            self.document_mappings = {
                doc_id: self.som.winner(embedding) for doc_id, embedding in zip(document_ids, embeddings)
            }

    def get_document_position(self, embedding: np.ndarray) -> Tuple[int, int]:
        """Get position for a document by its embedding"""
        return self.som.winner(embedding)
