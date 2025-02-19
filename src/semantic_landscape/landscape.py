from typing import Dict, List, Optional, Tuple

import numpy as np

from .encoder import SemanticEncoder
from .llm import LLMInterface
from .som import InteractiveSOM


class SemanticLandscape:
    """Main class coordinating the Interactive Semantic Landscape system"""

    def __init__(self, grid_size: Tuple[int, int] = (20, 20), use_hdbscan=False, min_cluster_size=5):
        """Initialize with basic components"""
        self.encoder = SemanticEncoder()
        self.documents: Dict[str, str] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.som: Optional[InteractiveSOM] = None
        self.grid_size = grid_size
        self.cluster_summaries = {}
        self.llm = LLMInterface()
        self.user_preferences: Dict[str, List[str]] = {}
        self.use_hdbscan = use_hdbscan
        self.min_cluster_size = min_cluster_size

    def add_documents(self, documents: List[str]) -> None:
        """Add documents to the landscape and train the SOM."""
        # Ensure documents is a list of strings
        if not isinstance(documents, list):
            documents = list(documents)

        # Get embeddings for new documents
        new_embeddings = self.encoder.encode(documents)

        # Generate document IDs
        doc_ids = [str(len(self.documents) + i) for i in range(len(documents))]

        # Combine new embeddings with existing ones
        all_embeddings = []
        all_doc_ids = []

        # Add existing embeddings
        if self.embeddings:
            all_embeddings.extend(list(self.embeddings.values()))
            all_doc_ids.extend(list(self.embeddings.keys()))

        # Add new embeddings
        all_embeddings.extend(new_embeddings)
        all_doc_ids.extend(doc_ids)

        all_embeddings = np.array(all_embeddings)

        # Initialize or update SOM
        if self.som is None:
            self.som = InteractiveSOM(
                input_dim=new_embeddings.shape[1],
                grid_size=self.grid_size,
                use_hdbscan=self.use_hdbscan,
                min_cluster_size=self.min_cluster_size,
            )
            self.som.som.random_weights_init(all_embeddings)

        # Train SOM with all document IDs
        self.som.train(all_embeddings, document_ids=all_doc_ids)

        # Store new documents and their embeddings
        for i, (doc, doc_id) in enumerate(zip(documents, doc_ids)):
            self.documents[doc_id] = doc
            self.embeddings[doc_id] = new_embeddings[i]

    def add_user_preference(
        self,
        preference_id: str,
        document_ids: List[str],
        target_position: Optional[Tuple[int, int]] = None,
        neighborhood_effect: float = 1.0,
    ):
        """Add a user preference to move documents to a target position.

        Args:
            preference_id: Unique identifier for this preference
            document_ids: List of document IDs to move
            target_position: Target (x,y) position on grid. If None, will be automatically determined.
            neighborhood_effect: Controls how much the movement affects other documents (0-1)
                               0 = no effect on others, 1 = full effect (default)
        """
        if not document_ids:
            return

        # Move each document individually to maintain the target position
        for doc_id in document_ids:
            if doc_id in self.embeddings:
                self.som.force_document_position(
                    doc_id, self.embeddings[doc_id], target_position, neighborhood_effect=neighborhood_effect
                )

        # Refresh with fewer epochs to stabilize while maintaining positions
        self.refresh(epochs=5)

    def move_document(self, doc_id: str, target_position: Tuple[int, int], preference_weight: float = 0.5):
        """
        Suggest a new position for a document by training the SOM weights

        Args:
            doc_id: ID of the document to move
            target_position: Suggested target position on the SOM grid
            preference_weight: How strongly to weight the suggestion (via learning rate)
        """
        if doc_id not in self.embeddings or self.som is None:
            raise ValueError("Document not found or SOM not initialized")

        # Get the document's embedding
        embedding = self.embeddings[doc_id]

        # Update the SOM weights to try to accommodate the suggested position
        self.som.force_document_position(doc_id, embedding, target_position)

    def get_document_positions(self) -> Dict[str, Tuple[int, int]]:
        """Get current positions of all documents on the SOM"""
        if self.som is None:
            return {}

        positions = {}
        for doc_id, embedding in self.embeddings.items():
            # Use the document ID to get position
            positions[doc_id] = self.som.get_position_by_id(doc_id)

        return positions

    def refresh(self, epochs: int = 10):
        """
        Refresh the SOM by retraining on the current embeddings.
        This allows the map to adapt while maintaining relative document positions.
        """
        if self.som:
            # Get current document positions
            current_positions = {doc_id: self.som.get_position_by_id(doc_id) for doc_id in self.embeddings.keys()}

            # Calculate centroids for each position
            position_centroids = {}
            for pos in set(current_positions.values()):
                docs_at_pos = [doc_id for doc_id, p in current_positions.items() if p == pos]
                embeddings_at_pos = [self.embeddings[doc_id] for doc_id in docs_at_pos]
                position_centroids[pos] = np.mean(embeddings_at_pos, axis=0)

            # Retrain with reduced learning rate to maintain stability
            embeddings = np.array(list(self.embeddings.values()))
            document_ids = list(self.embeddings.keys())

            # First train with original embeddings
            self.som.train(embeddings, epochs=epochs // 2, document_ids=document_ids)

            # Then train with centroids to maintain cluster structure
            for pos, centroid in position_centroids.items():
                docs = [doc_id for doc_id, p in current_positions.items() if p == pos]
                if docs:  # Only update if there are documents at this position
                    # Train with higher learning rate on centroid
                    self.som.force_document_position(docs[0], centroid, pos)

            # Final training pass to stabilize
            self.som.train(embeddings, epochs=epochs // 2, document_ids=document_ids)

    def get_cluster_summary(self, position: Tuple[int, int]) -> str:
        """Get or generate a summary for documents in a cluster at the given position."""
        if position not in self.cluster_summaries:
            # Get documents in this cluster
            cluster_docs = [doc_id for doc_id, pos in self.get_document_positions().items() if pos == position]

            if not cluster_docs:
                return "Empty cluster"

            # Get the text content of documents in this cluster
            texts = [self.documents[doc_id] for doc_id in cluster_docs]

            # Use LLM to generate a meaningful summary
            try:
                summary = self.llm.summarize_cluster(texts)
            except Exception as e:
                # Fallback to simple summary if LLM fails
                summary = texts[0][:100] + "..." if len(texts[0]) > 100 else texts[0]

            self.cluster_summaries[position] = summary

        return self.cluster_summaries[position]

    def suggest_splits(self) -> Dict[Tuple[int, int], Dict[str, List[str]]]:
        """Analyze all clusters and suggest potential splits"""
        suggestions = {}

        # First, get all unique positions
        all_positions = set(self.som.document_mappings.values())

        for pos in all_positions:
            # Get documents in this cluster
            docs = self._get_cluster_documents(pos)
            if len(docs) >= 2:  # Changed from 3 to 2 - we can split if there are at least 2 docs
                # Get the actual document contents
                doc_texts = [self.documents[d] for d in docs]

                # Try to get split suggestion from LLM
                split = self.llm.suggest_cluster_split(doc_texts)
                if split:
                    # Map the split back to document IDs
                    split_with_ids = {
                        "cluster1": [doc_id for doc_id in docs if self.documents[doc_id] in split["cluster1"]],
                        "cluster2": [doc_id for doc_id in docs if self.documents[doc_id] in split["cluster2"]],
                    }
                    if split_with_ids["cluster1"] and split_with_ids["cluster2"]:  # Only add if we got a real split
                        suggestions[pos] = split_with_ids

        return suggestions

    def _get_cluster_documents(self, position: Tuple[int, int]) -> List[str]:
        """Get document IDs belonging to a cluster at given position"""
        return [doc_id for doc_id, pos in self.som.document_mappings.items() if pos == position]

    def _update_cluster_summaries(self, positions: List[Tuple[int, int]]):
        """Update summaries for specified clusters"""
        for pos in positions:
            if pos in self.cluster_summaries:
                del self.cluster_summaries[pos]  # Force regeneration

    def answer_cluster_question(self, position: Tuple[int, int], question: str) -> str:
        """
        Answer a clarifying question about a particular cluster.

        Args:
            position: The position (cluster) on the SOM.
            question: The question to answer about the cluster.

        Returns:
            The answer from the LLM.
        """
        docs = self._get_cluster_documents(position)
        if not docs:
            return "Cluster is empty."
        return self.llm.answer_question(question, [self.documents[d] for d in docs])

    def apply_split(self, position: Tuple[int, int], split: Dict[str, List[str]]) -> None:
        """
        Apply a suggested split by moving documents to new positions.

        Args:
            position: Original cluster position
            split: Dictionary containing 'cluster1' and 'cluster2' document IDs
        """
        if not self.som:
            return

        # Find two available positions near the original cluster
        original_x, original_y = position
        positions = [
            (original_x + 1, original_y),  # right
            (original_x - 1, original_y),  # left
            (original_x, original_y + 1),  # down
            (original_x, original_y - 1),  # up
        ]

        # Filter valid positions within grid
        valid_positions = [
            pos for pos in positions if 0 <= pos[0] < self.grid_size[0] and 0 <= pos[1] < self.grid_size[1]
        ][
            :2
        ]  # Take first two valid positions

        if len(valid_positions) < 2:
            raise ValueError("Not enough space around cluster for splitting")

        # Move documents to new positions
        for i, cluster_docs in enumerate([split["cluster1"], split["cluster2"]]):
            for doc_id in cluster_docs:
                self.move_document(doc_id, valid_positions[i])

        # Retrain SOM to stabilize
        self.refresh(epochs=5)

    def suggest_merges(self) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Analyze clusters and suggest potential merges based on similarity"""
        if not self.som:
            return []

        suggestions = []
        positions = set(self.som.document_mappings.values())

        # Compare each pair of clusters
        for pos1 in positions:
            docs1 = self._get_cluster_documents(pos1)
            if not docs1:
                continue

            for pos2 in positions:
                if pos1 >= pos2:  # Skip self-comparisons and duplicates
                    continue

                docs2 = self._get_cluster_documents(pos2)
                if not docs2:
                    continue

                # Get texts for both clusters
                texts1 = [self.documents[d] for d in docs1]
                texts2 = [self.documents[d] for d in docs2]

                # Ask LLM if clusters should be merged
                should_merge = self.llm.suggest_merge(texts1, texts2)
                if should_merge:
                    suggestions.append((pos1, pos2))

        return suggestions

    def apply_merge(self, positions: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
        """
        Merge two clusters by moving all documents to one position.

        Args:
            positions: Tuple of two cluster positions to merge
        """
        if not self.som:
            return

        pos1, pos2 = positions

        # Choose target position (use first position)
        target_pos = pos1

        # Get documents from second cluster
        docs_to_move = self._get_cluster_documents(pos2)

        # Move all documents from second cluster to first cluster
        for doc_id in docs_to_move:
            self.move_document(doc_id, target_pos)

        # Retrain SOM to stabilize
        self.refresh(epochs=5)
