from semantic_landscape import SemanticLandscape

# Initialize with a smaller grid
landscape = SemanticLandscape(grid_size=(10, 10))

# Add fewer, more distinct documents
documents = [
    "Machine learning is a field of artificial intelligence.",
    "Python is a programming language.",
    "Data science involves statistics and programming.",
]

landscape.add_documents(documents)

# Get initial positions
positions = landscape.get_document_positions()
print("\nInitial positions:", positions)

# Try to move document '0' to center of grid (5,5)
print("\nAttempting to move document '0' to position (2, 2)...")
landscape.move_document("0", target_position=(2, 2))
print("After move:", landscape.get_document_positions())

# Refresh once with more epochs to stabilize
print("\nRefreshing SOM...")
landscape.refresh(epochs=50)  # Increased epochs for better convergence
print("Final positions:", landscape.get_document_positions())
