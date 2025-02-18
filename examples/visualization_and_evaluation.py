import numpy as np

from semantic_landscape import SemanticLandscape
from semantic_landscape.evaluation.metrics import ClusterEvaluator
from semantic_landscape.visualization.grid import SOMVisualizer

# Initialize the system
landscape = SemanticLandscape(grid_size=(10, 10))

# Add some example documents
documents = {
    "doc1": "Machine learning is a subset of artificial intelligence.",
    "doc2": "Neural networks are used in deep learning.",
    "doc3": "Python is a popular programming language.",
    "doc4": "JavaScript is used for web development.",
    "doc5": "TensorFlow is a machine learning framework.",
    "doc6": "React is a JavaScript framework for web development.",
}

# Add documents to the system
landscape.add_documents(documents)

# Get document positions
positions = landscape.get_document_positions()

# Initialize visualizer
visualizer = SOMVisualizer(grid_size=(10, 10))

# Get cluster summaries for labels
cluster_labels = {}
for pos in set(positions.values()):
    cluster_labels[pos] = landscape.get_cluster_summary(pos)

# Create visualization
fig = visualizer.plot_som_grid(
    weights=landscape.som.som.get_weights(), document_positions=positions, cluster_labels=cluster_labels
)

# Save or show the plot
fig.write_html("som_visualization.html")

# Evaluate clustering
evaluator = ClusterEvaluator(np.array(list(landscape.embeddings.values())))

# Convert positions to clusters
clusters = {}
for pos in set(positions.values()):
    clusters[pos] = [i for i, p in enumerate(positions.values()) if p == pos]

# Calculate metrics
coherence = evaluator.calculate_cluster_coherence(clusters)

# Check if we have valid number of clusters for separation calculation
n_samples = len(landscape.embeddings)
n_clusters = len(clusters)

print("\nClustering Evaluation:")
print(f"Average Coherence: {np.mean(list(coherence.values())):.3f}")

if 2 <= n_clusters < n_samples:
    separation = evaluator.calculate_cluster_separation(clusters)
    silhouette = evaluator.calculate_silhouette(clusters)
    print(f"Cluster Separation: {separation:.3f}")
    print(f"Silhouette Score: {silhouette:.3f}")
else:
    print("Note: Separation and Silhouette scores require 2 or more clusters")
    print(f"Current number of clusters: {n_clusters}")
    print(f"Number of samples: {n_samples}")
