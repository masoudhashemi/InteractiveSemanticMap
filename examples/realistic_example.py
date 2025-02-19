import math

import pandas as pd

from semantic_landscape import SemanticLandscape

# Load the dataset using pandas
df = pd.read_json("hf://datasets/mteb/twentynewsgroups-clustering/test.jsonl", lines=True)

documents = df["sentences"].tolist()[0]
labels = df["labels"].tolist()[0]

# Determine the number of unique labels
num_labels = len(set(labels))

# Set grid size to be approximately the square root of the number of labels
grid_size = (math.ceil(math.sqrt(num_labels)), math.ceil(math.sqrt(num_labels)))

landscape = SemanticLandscape(grid_size=grid_size, use_hdbscan=True, min_cluster_size=3)
print("Adding documents to the landscape...")
landscape.add_documents(documents)
positions = landscape.get_document_positions()

# Example of adding a user preference
example_doc_id = list(positions.keys())[0]
example_index = int(example_doc_id)  # Convert to integer index
current_position = positions[example_doc_id]
example_label = labels[example_index]

# Find all documents with the same label
same_label_doc_ids = [doc_id for doc_id, label in enumerate(labels) if label == example_label]

# Select the first 6 documents to move
few_doc_ids_to_move = same_label_doc_ids[:6]


# Define a new position that is different from the current one
def calculate_new_position(current_position, grid_size):
    # Move to the opposite corner of the grid
    new_x = (current_position[0] + 1) % grid_size[0]
    new_y = (current_position[1] + 1) % grid_size[1]
    return (new_x, new_y)


new_position = calculate_new_position(current_position, landscape.grid_size)

# Show initial positions
print("\n=== Initial Positions ===")
for doc_id in few_doc_ids_to_move:
    doc_id_str = str(doc_id)  # Convert to string to match keys in positions
    print(f"Document ID: {doc_id}, Initial Position: {positions[doc_id_str]}")

# Attempt to move documents with different neighborhood effects
print("\n=== Testing Different Neighborhood Effects ===")
print(f"Current Position: {current_position}, Target Position: {new_position}")

# Try with low neighborhood effect
print("\nMoving with low neighborhood effect (0.3)...")
landscape.add_user_preference(
    "Example_Category_Minimal",
    [str(doc_id) for doc_id in few_doc_ids_to_move],
    target_position=new_position,
    neighborhood_effect=0.3,
)

# Show intermediate positions
intermediate_positions = landscape.get_document_positions()
print("\nPositions after minimal effect move:")
for doc_id in few_doc_ids_to_move:
    doc_id_str = str(doc_id)
    print(f"Document ID: {doc_id}, Position: {intermediate_positions[doc_id_str]}")

# Try with high neighborhood effect
print("\nMoving with high neighborhood effect (1.0)...")
for _ in range(5):
    landscape.add_user_preference(
        "Example_Category_Strong",
        [str(doc_id) for doc_id in few_doc_ids_to_move],
        target_position=new_position,
        neighborhood_effect=1.0,
    )

# Show final positions
final_positions = landscape.get_document_positions()

# Show which documents moved and by how much
print("\n=== Document Movements Summary ===")
print("\nInitial → Minimal Effect → Strong Effect:")
for doc_id in few_doc_ids_to_move:
    doc_id_str = str(doc_id)
    print(f"Document {doc_id}:")
    print(f"  {positions[doc_id_str]} → {intermediate_positions[doc_id_str]} → {final_positions[doc_id_str]}")

# Show final clusters
# print("\n=== Final Clusters ===")
# new_clusters = {}
# for doc_id, pos in new_positions.items():
#     if pos not in new_clusters:
#         new_clusters[pos] = []
#     new_clusters[pos].append(doc_id)

# for pos, doc_ids in new_clusters.items():
#     print(f"\nFinal Cluster at {pos}:")
#     print("Documents:")
#     for doc_id in doc_ids:
#         print(f"- {landscape.documents[doc_id]}")
#     print(f"Summary: {landscape.get_cluster_summary(pos)}")
