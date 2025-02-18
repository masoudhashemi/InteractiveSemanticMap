from typing import Dict, List, Optional

import numpy as np
from litellm import completion


class LLMInterface:
    """Handles LLM interactions for cluster analysis and feedback interpretation"""

    def __init__(self, model: str = "gpt-4o-mini"):
        """
        Initialize LLM interface

        Args:
            model: Model identifier to use with LiteLLM
        """
        self.model = model

    def _get_completion(self, prompt: str) -> str:
        """Helper method to get LLM completion"""
        try:
            response = completion(model=self.model, messages=[{"role": "user", "content": prompt}])
            content = response.choices[0].message.content
            return content
        except Exception as e:
            print(f"Error getting LLM completion: {e}")
            raise

    def summarize_cluster(self, documents: List[str]) -> str:
        """
        Generate a summary for a cluster of documents

        Args:
            documents: List of document texts in the cluster

        Returns:
            A concise summary of the cluster's theme
        """
        prompt = "Summarize the common theme of these text excerpts in a single sentence:\n\n" f"{'\n'.join(documents)}"
        return self._get_completion(prompt)

    def compare_documents(self, doc1: str, doc2: str) -> float:
        """
        Compare two documents and return a similarity score

        Args:
            doc1: First document text
            doc2: Second document text

        Returns:
            Similarity score between 0 and 1
        """
        prompt = (
            "On a scale of 1 to 10, how similar are these two documents in terms "
            "of their topic and meaning? Respond with just the number.\n\n"
            f"Document 1: {doc1}\n"
            f"Document 2: {doc2}"
        )

        score = float(self._get_completion(prompt))
        return score / 10.0  # Normalize to 0-1

    def suggest_cluster_split(self, documents: List[str]) -> Optional[Dict[str, List[str]]]:
        """
        Suggest how to split a cluster of documents based on their content

        Args:
            documents: List of document texts in the cluster

        Returns:
            Dictionary with 'cluster1' and 'cluster2' lists of documents, or None if no split needed
        """
        if len(documents) < 2:
            return None

        prompt = (
            "Analyze these documents and split them into two groups based on their topics "
            "or themes. If they are too similar to split, respond with 'NO_SPLIT'.\n\n"
            "Documents:\n" + "\n".join(f"{i+1}. {doc}" for i, doc in enumerate(documents)) + "\n\n"
            "Respond in the format:\nGroup 1: [list document numbers]\nGroup 2: [list document numbers]\n"
            "Or just respond 'NO_SPLIT' if documents are too similar."
        )

        response = self._get_completion(prompt)

        print("Response: ", response)

        if "NO_SPLIT" in response.upper():
            return None

        try:
            # Parse the response to get document groupings
            group1_docs = []
            group2_docs = []

            for line in response.split("\n"):
                if line.startswith("Group 1:"):
                    numbers = [int(n.strip()) for n in line.replace("Group 1:", "").strip().strip("[]").split(",")]
                    group1_docs = [documents[i - 1] for i in numbers]
                elif line.startswith("Group 2:"):
                    numbers = [int(n.strip()) for n in line.replace("Group 2:", "").strip().strip("[]").split(",")]
                    group2_docs = [documents[i - 1] for i in numbers]

            if group1_docs and group2_docs:
                return {"cluster1": group1_docs, "cluster2": group2_docs}
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return None

        return None

    def interpret_feedback(
        self, moved_doc: str, source_cluster_docs: List[str], target_cluster_docs: List[str]
    ) -> float:
        """
        Interpret user feedback when moving a document between clusters

        Args:
            moved_doc: The document being moved
            source_cluster_docs: Documents in the source cluster
            target_cluster_docs: Documents in the target cluster

        Returns:
            A confidence score (0-1) for the move
        """
        prompt = (
            "A user has moved a document from one cluster to another. "
            "On a scale of 1-10, how confident are you that this move makes sense? "
            "Respond with just the number.\n\n"
            f"Moved document: {moved_doc}\n\n"
            f"Original cluster documents:\n{'\n'.join(source_cluster_docs)}\n\n"
            f"Target cluster documents:\n{'\n'.join(target_cluster_docs)}"
        )

        confidence = float(self._get_completion(prompt))
        return confidence / 10.0

    def answer_question(self, question: str, documents: List[str]) -> str:
        """
        Answer a clarifying question based on a cluster of documents.

        Args:
            question: The question to answer.
            documents: The list of document texts in the cluster.

        Returns:
            The answer from the LLM.
        """
        prompt = (
            f"Based on the following documents, answer the question:\n\n"
            f"{'\n'.join(documents)}\n\n"
            f"Question: {question}"
        )
        return self._get_completion(prompt)

    def suggest_merge(self, cluster1_docs: List[str], cluster2_docs: List[str]) -> bool:
        """
        Suggest whether two clusters should be merged based on their content.

        Args:
            cluster1_docs: List of document texts in first cluster
            cluster2_docs: List of document texts in second cluster

        Returns:
            Boolean indicating whether clusters should be merged
        """
        prompt = (
            "Analyze these two groups of documents and determine if they are similar enough "
            "to be merged into a single cluster. Respond with just 'YES' or 'NO'.\n\n"
            "Group 1:\n" + "\n".join(f"- {doc}" for doc in cluster1_docs) + "\n\n"
            "Group 2:\n" + "\n".join(f"- {doc}" for doc in cluster2_docs)
        )

        response = self._get_completion(prompt).strip().upper()
        return response == "YES"
