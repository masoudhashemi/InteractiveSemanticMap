from semantic_landscape import SemanticLandscape
from semantic_landscape.som import InteractiveSOM

# Initialize the system
landscape = SemanticLandscape(
    grid_size=(10, 10), use_hdbscan=True, min_cluster_size=5
)  # Smaller grid for clearer clusters

# Add documents that naturally form distinct groups
documents = [
    # ML/AI Group
    "Deep learning models use neural networks for complex pattern recognition.",
    "Machine learning algorithms learn patterns from training data.",
    "Artificial intelligence systems can make autonomous decisions.",
    # Web Dev Group
    "HTML and CSS are fundamental technologies for web development.",
    "JavaScript frameworks like React improve web application development.",
    "Web developers use Node.js for server-side programming.",
    # Data Science Group
    "Data scientists use Python for statistical analysis.",
    "Data visualization helps understand complex datasets.",
    "SQL is essential for database management in data science.",
]

# Add documents and get their IDs
landscape.add_documents(documents)

# Get document positions and cluster summaries
positions = landscape.get_document_positions()
print("\nInitial document positions and summaries:")
for doc_id, pos in positions.items():
    summary = landscape.get_cluster_summary(pos)
    print(f"\nDocument {doc_id} at position {pos}")
    print(f"Content: {landscape.documents[doc_id]}")
    print(f"Cluster Summary from LLM: {summary}")

# Ask some questions about specific clusters
print("\nAsking questions about clusters:")
for pos in set(positions.values()):
    question = "What is the main topic of this cluster?"
    answer = landscape.answer_cluster_question(pos, question)
    print(f"\nCluster at position {pos}:")
    print(f"Question: {question}")
    print(f"LLM Answer: {answer}")

# Check for suggested splits with detailed output
print("\nChecking for potential cluster splits...")
splits = landscape.suggest_splits()
for pos, split in splits.items():
    print(f"\nSuggested split for cluster at {pos}:")
    print("\nCluster 1 documents:")
    for doc_id in split["cluster1"]:
        print(f"- {landscape.documents[doc_id]}")
    print("\nCluster 2 documents:")
    for doc_id in split["cluster2"]:
        print(f"- {landscape.documents[doc_id]}")

# Check for suggested merges
print("\nChecking for potential cluster merges...")
merge_suggestions = landscape.suggest_merges()
for pos1, pos2 in merge_suggestions:
    print(f"\nSuggested merge between clusters at {pos1} and {pos2}:")
    print("\nCluster 1 documents:")
    for doc_id in landscape._get_cluster_documents(pos1):
        print(f"- {landscape.documents[doc_id]}")
    print("\nCluster 2 documents:")
    for doc_id in landscape._get_cluster_documents(pos2):
        print(f"- {landscape.documents[doc_id]}")

    # Apply the merge if desired
    landscape.apply_merge((pos1, pos2))
    print(f"\nMerged clusters at {pos1} and {pos2}")

# If we have any splits to apply
if splits:
    first_split = next(iter(splits.items()))
    position, split_docs = first_split
    print(f"\nApplying split for cluster at {position}")
    landscape.apply_split(position, split_docs)
    print("Split applied")

# Try to move a document and show the impact
first_doc_id = list(positions.keys())[0]
original_pos = positions[first_doc_id]
new_pos = (5, 5)

print(f"\nMoving document {first_doc_id} from {original_pos} to {new_pos}...")
print(f"Original document: {landscape.documents[first_doc_id]}")

# Get original cluster summary
original_summary = landscape.get_cluster_summary(original_pos)
print(f"Original cluster summary: {original_summary}")

# Move the document
landscape.move_document(first_doc_id, target_position=new_pos)

# Get new cluster summary
new_summary = landscape.get_cluster_summary(new_pos)
print(f"New cluster summary at {new_pos}: {new_summary}")

# Show final positions
print("\nFinal document positions:", landscape.get_document_positions())
