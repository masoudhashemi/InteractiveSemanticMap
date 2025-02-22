from typing import List, Optional, Tuple, Union

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
        self,
        doc_ids: Union[str, List[str]],
        embeddings: Union[np.ndarray, List[np.ndarray]],
        target_positions: Union[Tuple[int, int], List[Tuple[int, int]]],
        neighborhood_effect: float = 1.0,
    ):
        """
        Force documents to specific positions while maintaining semantic relationships.
        Similar documents will be influenced to move together.

        Args:
            doc_ids: Single document ID or list of document IDs to move
            embeddings: Single embedding or list of embeddings
            target_positions: Single target position or list of target positions as (x,y) tuples
            neighborhood_effect: Controls how much the movement affects similar documents (0-1)
        """
        # Convert single inputs to lists
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
            embeddings = [embeddings]
            target_positions = [target_positions]

        # Validate inputs
        if not (len(doc_ids) == len(embeddings) == len(target_positions)):
            raise ValueError("Number of documents, embeddings, and target positions must match")

        for target_pos in target_positions:
            if not (0 <= target_pos[0] < self.grid_size[0] and 0 <= target_pos[1] < self.grid_size[1]):
                raise ValueError("Target position is out of grid bounds")

        if not 0 <= neighborhood_effect <= 1:
            raise ValueError("neighborhood_effect must be between 0 and 1")

        # Store document data
        for doc_id, embedding in zip(doc_ids, embeddings):
            self.document_data[doc_id] = embedding

        # Initialize parameters
        sigma = 2.0
        learning_rate = 0.5
        max_iterations = 10

        # Get all document embeddings for similarity calculations
        all_embeddings = np.array(list(self.document_data.values()))
        all_doc_ids = list(self.document_data.keys())

        for _ in range(max_iterations):
            # Process each target document
            for doc_id, embedding, target_pos in zip(doc_ids, embeddings, target_positions):
                current_pos = self.som.winner(embedding)

                # Calculate similarity scores with all other documents
                similarities = np.dot(all_embeddings, embedding) / (
                    np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(embedding)
                )

                # Calculate movement vector
                movement = np.array(target_pos) - np.array(current_pos)
                movement_direction = movement / np.linalg.norm(movement) if np.any(movement) else np.zeros(2)

                # Update weights for target document and similar documents
                for i in range(self.grid_size[0]):
                    for j in range(self.grid_size[1]):
                        # Calculate base influence from target position
                        dist_to_target = np.sqrt((i - target_pos[0]) ** 2 + (j - target_pos[1]) ** 2)
                        position_influence = np.exp(-dist_to_target / sigma)

                        # Add directional influence
                        node_direction = np.array([i - current_pos[0], j - current_pos[1]])
                        if np.any(node_direction):
                            node_direction = node_direction / np.linalg.norm(node_direction)
                            direction_alignment = np.dot(movement_direction, node_direction)
                            position_influence *= max(0, direction_alignment)

                        # Find documents currently mapped to this position
                        docs_at_pos = [
                            (idx, sim)
                            for idx, (did, sim) in enumerate(zip(all_doc_ids, similarities))
                            if self.som.winner(all_embeddings[idx]) == (i, j)
                        ]

                        if docs_at_pos:
                            # Calculate semantic influence based on document similarities
                            max_similarity = max(sim for _, sim in docs_at_pos)
                            semantic_influence = max_similarity * neighborhood_effect
                        else:
                            semantic_influence = 0

                        # Combine position and semantic influence
                        total_influence = max(position_influence, semantic_influence)

                        # Calculate weight update
                        if (i, j) == target_pos:
                            # Direct update for target position
                            self.som._weights[i, j] = embedding
                        else:
                            # Update based on combined influence
                            weight_delta = embedding - self.som._weights[i, j]
                            self.som._weights[i, j] += learning_rate * total_influence * weight_delta

                # Update positions of all documents
                for d_id, d_embedding in self.document_data.items():
                    self.document_mappings[d_id] = self.som.winner(d_embedding)

            # Check if target documents reached their positions
            all_reached = all(
                self.som.winner(emb) == target_pos for emb, target_pos in zip(embeddings, target_positions)
            )
            if all_reached:
                break

            # Reduce parameters
            learning_rate *= 0.95
            sigma *= 0.95

        # Final position enforcement for target documents
        for doc_id, embedding, target_pos in zip(doc_ids, embeddings, target_positions):
            self.som._weights[target_pos[0], target_pos[1]] = embedding
            self.document_mappings[doc_id] = target_pos

        return True

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
