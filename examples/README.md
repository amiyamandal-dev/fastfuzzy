# FastFuzzy Examples

This directory contains practical examples demonstrating various use cases for the fastfuzzy library.

## Examples Overview

### 1. Basic Fuzzy Matcher Demo (`fuzzy_matcher_demo.py`)
Demonstrates core fuzzy matching capabilities:
- Basic string similarity functions
- Fuzzy search in lists
- Batch processing
- Phonetic matching
- Performance comparisons
- Custom scorers

**Run with:**
```bash
python fuzzy_matcher_demo.py
```

### 2. Data Deduplication (`data_deduplication.py`)
Shows how to use fuzzy matching for identifying and removing duplicate records:
- Record similarity calculation
- Duplicate detection
- Clustering similar records
- Merging duplicate clusters

**Run with:**
```bash
python data_deduplication.py
```

### 3. Spell Checker (`spell_checker.py`)
Implementation of a spell checker using fuzzy matching:
- Word validation
- Spelling corrections
- Text checking
- Sentence suggestions
- Performance optimization

**Run with:**
```bash
python spell_checker.py
```

### 4. Fuzzy API Service (`fuzzy_api_service.py`)
Web API service built with FastAPI for fuzzy matching operations:
- RESTful endpoints for similarity calculations
- Batch processing API
- Fuzzy search endpoint
- Text processing utilities

**Run with:**
```bash
uvicorn fuzzy_api_service:app --reload
```

API documentation available at: http://localhost:8000/docs

### 5. Log Analysis (`log_analysis.py`)
Log analysis tool using fuzzy matching to identify patterns:
- Similar message grouping
- Pattern identification
- Anomaly detection
- Custom analysis scenarios

**Run with:**
```bash
python log_analysis.py
```

## Installation

To run these examples, you'll need to install the required dependencies:

```bash
# Install fastfuzzy (from project root)
maturin develop --release

# Install additional dependencies for examples
pip install fastapi uvicorn
```

## Usage

1. **Build the fastfuzzy library** (from project root):
   ```bash
   maturin develop --release
   ```

2. **Navigate to the examples directory**:
   ```bash
   cd examples
   ```

3. **Run any example**:
   ```bash
   python fuzzy_matcher_demo.py
   ```

## Example Applications

### Web API Service
The fuzzy API service provides a RESTful interface for all fastfuzzy capabilities:

- **POST /similarity** - Calculate similarity between two strings
- **POST /batch-similarity** - Calculate similarity matrix for batches
- **POST /fuzzy-search** - Perform fuzzy search in a list of choices
- **POST /text-process** - Process text using various utilities
- **GET /algorithms** - List available algorithms
- **GET /examples** - Get example requests

### Real-world Use Cases

1. **Data Cleaning** - Remove duplicates from customer databases
2. **Log Analysis** - Group similar error messages for better monitoring
3. **Spell Checking** - Provide intelligent spelling corrections
4. **Search Enhancement** - Improve search results with fuzzy matching
5. **Record Linkage** - Match records across different databases

## Performance Tips

1. **Use batch functions** for multiple comparisons
2. **Set appropriate score cutoffs** to skip unnecessary calculations
3. **Choose the right algorithm** for your use case:
   - `ratio`: General purpose
   - `partial_ratio`: Substring matching
   - `token_sort_ratio`: Word order doesn't matter
   - `token_set_ratio`: Duplicate words don't matter
   - `wratio`: Automatic selection

4. **Use caching wisely** - The library includes intelligent caching
5. **Profile your usage** - Monitor performance with built-in tools

## Customization

Each example can be easily customized for your specific needs:

- Adjust similarity thresholds
- Implement custom scoring algorithms
- Add new text preprocessing steps
- Extend API endpoints
- Modify data structures for your use case

## Contributing

Feel free to extend these examples or create new ones for additional use cases!