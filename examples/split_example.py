from semantic_landscape import SemanticLandscape

# Initialize the landscape with a smaller grid for clearer visualization
landscape = SemanticLandscape(grid_size=(2, 2), use_hdbscan=True, min_cluster_size=3)

# Create documents that naturally form clusters but with some ambiguity
documents = [
    # Programming Languages Cluster
    "Python is a popular programming language known for its readability.",
    "Java is widely used for enterprise software development.",
    "JavaScript is essential for web development and runs in browsers.",
    # Machine Learning Cluster
    "Neural networks are inspired by biological brain structures.",
    "Deep learning models require large amounts of training data.",
    "Supervised learning uses labeled data for training.",
    # Hybrid Document (could belong to either cluster)
    "Python's scikit-learn library provides machine learning tools for developers.",
]

# Add documents to the landscape
print("Adding documents to the landscape...")
landscape.add_documents(documents)

# Get initial positions and summaries
print("\n=== Initial State ===")
positions = landscape.get_document_positions()
for doc_id, pos in positions.items():
    print(f"\nDocument {doc_id} at position {pos}")
    print(f"Content: {landscape.documents[doc_id]}")
    print(f"Cluster Summary: {landscape.get_cluster_summary(pos)}")

# Show initial clusters more concisely
print("\n=== Initial Clusters ===")
clusters = {}
for doc_id, pos in positions.items():
    if pos not in clusters:
        clusters[pos] = []
    clusters[pos].append(doc_id)

for pos, doc_ids in clusters.items():
    print(f"\nCluster at {pos}:")
    print("Documents:")
    for doc_id in doc_ids:
        print(f"- {landscape.documents[doc_id]}")
    print(f"Summary: {landscape.get_cluster_summary(pos)}")

# Add a preference to move the hybrid document to the ML cluster
# First, find the hybrid document ID and ML cluster documents
hybrid_doc_id = None
ml_docs = []
for doc_id, doc in landscape.documents.items():
    if "scikit-learn" in doc:
        hybrid_doc_id = doc_id
    elif any(term in doc.lower() for term in ["neural", "deep learning", "supervised"]):
        ml_docs.append(doc_id)

# Find ML cluster position (using position of first ML document)
ml_cluster_pos = None
for doc_id in ml_docs:
    ml_cluster_pos = landscape.get_document_positions()[doc_id]
    break

# Add the preference
print("\n=== Adding Preference to Move Hybrid Document ===")
if ml_cluster_pos and hybrid_doc_id:
    landscape.add_user_preference("ML_tools", [hybrid_doc_id] + ml_docs, target_position=ml_cluster_pos)

# Retrain with preferences
print("Retraining with preferences...")
landscape.retrain_with_preferences()

# Show updated positions and summaries
print("\n=== Final State After Movement ===")
new_positions = landscape.get_document_positions()
for doc_id, pos in new_positions.items():
    print(f"\nDocument {doc_id} at position {pos}")
    print(f"Content: {landscape.documents[doc_id]}")
    print(f"Cluster Summary: {landscape.get_cluster_summary(pos)}")

# Show which documents moved
print("\n=== Document Movements ===")
moved_docs = [doc_id for doc_id in landscape.documents if positions[doc_id] != new_positions[doc_id]]
if moved_docs:
    print("\nMoved Documents:")
    for doc_id in moved_docs:
        print(f"- {landscape.documents[doc_id]}")
        print(f"  From: {positions[doc_id]} -> To: {new_positions[doc_id]}")

# Show final clusters
print("\n=== Final Clusters ===")
new_clusters = {}
for doc_id, pos in new_positions.items():
    if pos not in new_clusters:
        new_clusters[pos] = []
    new_clusters[pos].append(doc_id)

for pos, doc_ids in new_clusters.items():
    print(f"\nFinal Cluster at {pos}:")
    print("Documents:")
    for doc_id in doc_ids:
        print(f"- {landscape.documents[doc_id]}")
    print(f"Summary: {landscape.get_cluster_summary(pos)}")

# Ask LLM to explain the changes
print("\n=== Cluster Analysis ===")
for pos in set(new_positions.values()):
    print(f"\nAnalysis of Cluster at {pos}:")
    question = "How does this cluster's theme differ from its initial state?"
    answer = landscape.answer_cluster_question(pos, question)
    print(f"LLM Analysis: {answer}")
