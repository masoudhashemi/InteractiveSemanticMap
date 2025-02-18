from typing import Dict, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class SOMVisualizer:
    """Visualizer for Self-Organizing Map"""

    def __init__(self, grid_size: Tuple[int, int]):
        self.grid_size = grid_size

    def plot_som_grid(
        self,
        weights: np.ndarray,
        document_positions: Dict[str, Tuple[int, int]],
        cluster_labels: Dict[Tuple[int, int], str] = None,
    ) -> go.Figure:
        """
        Create interactive visualization of SOM grid

        Args:
            weights: SOM weight matrix
            document_positions: Mapping of document IDs to grid positions
            cluster_labels: Optional mapping of positions to cluster labels

        Returns:
            Plotly figure object
        """
        # Create figure with secondary y-axis
        fig = make_subplots(rows=1, cols=1)

        # Add heatmap of weights (using first principal component)
        heatmap = go.Heatmap(
            z=weights.mean(axis=2),  # Average across embedding dimensions
            colorscale="Viridis",
            showscale=True,
            name="Weight Distribution",
        )
        fig.add_trace(heatmap)

        # Add document markers
        x_coords = []
        y_coords = []
        text_labels = []

        for doc_id, pos in document_positions.items():
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            text_labels.append(doc_id)

        markers = go.Scatter(
            x=x_coords, y=y_coords, mode="markers+text", text=text_labels, textposition="top center", name="Documents"
        )
        fig.add_trace(markers)

        # Add cluster labels if provided
        if cluster_labels:
            label_x = []
            label_y = []
            label_text = []

            for pos, label in cluster_labels.items():
                label_x.append(pos[0])
                label_y.append(pos[1])
                label_text.append(label)

            labels = go.Scatter(
                x=label_x, y=label_y, mode="text", text=label_text, textposition="middle center", name="Cluster Labels"
            )
            fig.add_trace(labels)

        # Update layout
        fig.update_layout(title="SOM Document Clustering", xaxis_title="X", yaxis_title="Y", showlegend=True)

        return fig
