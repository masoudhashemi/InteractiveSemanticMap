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

    def force_document_position(
        self, doc_id: str, embedding: np.ndarray, target_position: Tuple[int, int], neighborhood_effect: float = 1.0
    ):
        """
        Force a document to a specific position by adjusting the SOM weights and neighborhood

        Args:
            doc_id: Identifier of the document to move
            embedding: Document embedding vector
            target_position: Desired (x,y) position on the grid
            neighborhood_effect: Controls how much the movement affects other documents (0-1)
                               0 = no effect on others, 1 = full effect (default)
        """
        # Ensure target position is within grid bounds
        if not (0 <= target_position[0] < self.grid_size[0] and 0 <= target_position[1] < self.grid_size[1]):
            raise ValueError("Target position is out of grid bounds")

        if not 0 <= neighborhood_effect <= 1:
            raise ValueError("neighborhood_effect must be between 0 and 1")

        # Store the document data
        self.document_data[doc_id] = embedding

        # Define parameters
        sigma = 2.0  # Neighborhood radius
        learning_rate = 0.5  # Learning rate
        max_iterations = 100

        # Calculate the target weights
        target_weights = embedding

        # Precompute the grid indices and distances
        x_indices, y_indices = np.meshgrid(np.arange(self.grid_size[0]), np.arange(self.grid_size[1]), indexing="ij")
        distances = np.sqrt((x_indices - target_position[0]) ** 2 + (y_indices - target_position[1]) ** 2)

        # Store original weights to blend between original and fully affected states
        original_weights = self.som._weights.copy()

        for _ in range(max_iterations):
            # Calculate neighborhood influence
            influence = np.exp(-(distances**2) / (2 * sigma**2))
            influence = influence.reshape(self.grid_size[0], self.grid_size[1], 1)

            # Update weights using neighborhood function
            for i in range(self.grid_size[0]):
                for j in range(self.grid_size[1]):
                    # Calculate weight update based on distance from target
                    weight_delta = target_weights - self.som._weights[i][j]
                    full_update = self.som._weights[i][j] + learning_rate * influence[i][j] * weight_delta

                    # Blend between original and fully affected weights based on neighborhood_effect
                    self.som._weights[i][j] = (
                        original_weights[i][j] * (1 - neighborhood_effect) + full_update * neighborhood_effect
                    )

            # Update positions of all documents
            for d_id, d_embedding in self.document_data.items():
                self.document_mappings[d_id] = self.som.winner(d_embedding)

            # Check if target document is in position
            current_position = self.som.winner(embedding)
            if current_position == target_position:
                break

            # Gradually reduce learning rate and neighborhood size
            learning_rate *= 0.95
            sigma *= 0.95

        # Use winner to get final position instead of direct assignment
        final_position = self.som.winner(embedding)
        self.document_mappings[doc_id] = final_position

        # Return whether we achieved the target position
        return final_position == target_position

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
