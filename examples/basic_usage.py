from semantic_landscape import SemanticLandscape

# Initialize a small landscape for clear visualization
landscape = SemanticLandscape(grid_size=(3, 3))

# Create some simple, related documents
documents = [
    "The cat sleeps on the mat",
    "The dog naps in its bed",
    "The kitten plays with yarn",
    "The puppy chases its tail",
    "The cat and dog play together",
]

# Add documents to the landscape
print("Adding documents to the landscape...")
landscape.add_documents(documents)

# Get initial positions
initial_positions = landscape.get_document_positions()

print("\n=== Initial Document Positions ===")
for doc_id, pos in initial_positions.items():
    print(f"Document: '{documents[int(doc_id)]}'")
    print(f"Position: {pos}")

# Let's try to move the "cat and dog" document (ID: 4)
# to a new position with different neighborhood effects
target_position = (2, 2)
doc_to_move = "3"

print("\n=== Moving Document with Different Neighborhood Effects ===")
print(f"Moving: '{documents[int(doc_to_move)]}'")
print(f"Target Position: {target_position}")

# First try with minimal neighborhood effect
print("\n1. Low Neighborhood Effect (0.2)")
print("This will mostly move just the target document")
landscape.add_user_preference("minimal_effect", [doc_to_move], target_position=target_position, neighborhood_effect=0.2)

# Show intermediate positions
intermediate_positions = landscape.get_document_positions()
print("\nPositions after minimal effect:")
for doc_id, pos in intermediate_positions.items():
    if pos != initial_positions[doc_id]:
        print(f"Document: '{documents[int(doc_id)]}' moved to {pos}")
    else:
        print(f"Document: '{documents[int(doc_id)]}' stayed at {pos}")

# Then try with strong neighborhood effect
print("\n2. High Neighborhood Effect (1.0)")
print("This will influence nearby related documents")
landscape.add_user_preference("strong_effect", [doc_to_move], target_position=target_position, neighborhood_effect=1.0)

# Show final positions
final_positions = landscape.get_document_positions()

print("\n=== Movement Summary ===")
print("Format: Initial → Low Effect → High Effect")
for doc_id in landscape.documents:
    print(f"\nDocument: '{documents[int(doc_id)]}'")
    print(f"Positions: {initial_positions[doc_id]} → {intermediate_positions[doc_id]} → {final_positions[doc_id]}")

# Show the effect on document relationships
print("\n=== Final Document Clusters ===")
clusters = {}
for doc_id, pos in final_positions.items():
    if pos not in clusters:
        clusters[pos] = []
    clusters[pos].append(documents[int(doc_id)])

for pos, docs in clusters.items():
    print(f"\nPosition {pos}:")
    for doc in docs:
        print(f"- {doc}")
