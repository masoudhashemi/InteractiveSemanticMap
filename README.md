# Semantic Landscape

A framework for creating interactive document landscapes that combine Self-Organizing Maps (SOM) with Large Language Models (LLMs) for intelligent document organization and analysis.

## Core Concepts

### Self-Organizing Map (SOM) Integration

- Documents are mapped to a 2D grid where proximity indicates semantic similarity
- Interactive weight adjustments allow for user-guided document organization
- Neighborhood functions ensure smooth transitions and maintain global structure
- Optional HDBSCAN initialization for better initial cluster formation

### LLM-Enhanced Analysis

- Intelligent cluster summarization and theme extraction
- Semantic similarity assessment for merge suggestions
- Context-aware cluster splitting recommendations
- Natural language Q&A about document clusters

### Interactive Learning

The system learns from user interactions through several mechanisms:

1. **Force-Based Positioning**
   - Documents can be manually positioned
   - Surrounding weights adjust to accommodate user preferences
   - Neighborhood influence ensures smooth transitions

2. **Preference Learning**
   - Users can specify document groups that should remain together
   - System maintains group cohesion during future updates
   - Weighted learning rates balance user input with semantic similarity

3. **Adaptive Retraining**
   - Periodic refresh maintains stability while incorporating new information
   - Centroid-based training preserves cluster structure
   - Multi-phase training process balances global and local organization

## Key Methods

### Document Organization

- `force_document_position`: Implements force-based document positioning with neighborhood adaptation
- `add_user_preference`: Captures and maintains user-defined document groupings
- `retrain_with_preferences`: Balances semantic similarity with user preferences

### Cluster Analysis

- `suggest_splits`: Uses LLM to identify semantically distinct subgroups within clusters
- `suggest_merges`: Analyzes cluster pairs for potential combination
- `get_cluster_summary`: Generates thematic summaries of document groups

### Interactive Features

- `answer_cluster_question`: Provides context-aware answers about document clusters
- `refresh`: Stabilizes the landscape while preserving learned relationships
- `move_document`: Implements user-guided document repositioning

## Architecture

The system consists of three main components:

1. **SOM Layer** (`InteractiveSOM`)
   - Handles document positioning and weight updates
   - Maintains neighborhood relationships
   - Implements force-based positioning

2. **LLM Layer** (`LLMInterface`)
   - Provides semantic analysis
   - Generates cluster insights
   - Evaluates potential cluster operations

3. **Coordination Layer** (`SemanticLandscape`)
   - Manages component interaction
   - Maintains document state
   - Implements high-level operations

## Technical Details

### SOM Implementation

- Customized MiniSom implementation
- Adaptive learning rates for stability
- Optional HDBSCAN initialization
- Neighborhood-aware weight updates

### LLM Integration

- Prompt engineering for consistent analysis
- Context-aware document comparison
- Structured output parsing
- Fallback mechanisms for reliability

### Document Processing

- Embedding-based representation
- Centroid-based cluster analysis
- Incremental updates
- Position-aware document management 