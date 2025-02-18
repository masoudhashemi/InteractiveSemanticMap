from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class SemanticEncoder:
    """Handles document encoding using Sentence-BERT"""

    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        """Initialize the encoder with a specific Sentence-BERT model"""
        self.model = SentenceTransformer(model_name)

    def encode(self, documents: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode documents into embeddings.

        Args:
            documents: List of strings to encode
            batch_size: Batch size for encoding

        Returns:
            numpy array of embeddings
        """
        if not documents:
            raise ValueError("No documents provided for encoding")

        # Ensure all inputs are strings
        documents = [str(doc) for doc in documents]

        # Encode documents
        embeddings = self.model.encode(documents, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True)

        return embeddings
