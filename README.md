# VectorDB from Scratch

<div style="display: flex; align-items: center; gap: 20px;">
  <img src="assets/logo.png" alt="VectorDB Logo" height="150" width="auto">
  <div>
    A from-scratch Vector Database implementation with REST API, featuring multiple indexing algorithms and MongoDB persistence.
  </div>
</div>

## Project Overview

This project implements a complete Vector Database system from scratch. The system supports multiple indexing algorithms, offers a RESTful API, and includes testing and documentation.

### Key Features

- **Multiple Index Types**: Flat, IVF (Inverted File), and LSH SimHash indexes
- **RESTful API**: Complete CRUD operations for libraries, documents, and chunks
- **Vector Search**: k-Nearest Neighbor similarity search
- **MongoDB Persistence**: Data persistence across container restarts
- **Docker Containerization**: Production-ready deployment
- **Comprehensive Testing**: end-to-end tests
- **Thread Safety**: Async read-write locks for concurrent access
- **Embedding Generation**: Integration with Cohere API for text embeddings via custom API call

## Architecture
<!-- TODO: add diagram -->
```
Client Layer → API Layer (FastAPI) → Service Layer → Storage Layer → Index Layer → Data Layer
```


## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.9+ (for local development)
- Cohere API key (for embedding generation)

### Running with Docker

1. **Clone the repository**
   ```bash
   git clone git@github.com:lucas-oma/vectorDB_from_scratch.git
   cd vectorDB_from_scratch
   ```

2. **Set up environment variables**
   ```bash
   cp env.example .env
   # Edit .env and add your Cohere API key
   ```
   For this step, only the cohere API is **mandatory**, other variables have default values assigned.

3. **Start the services**
   ```bash
   docker-compose up --build -d
   ```

4. **Verify the API is running**
   ```bash
   curl http://localhost:8000/v1/healthz
   ```

5. **Access the API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

6. **Test with Postman Collection**
   - Import `postman/VectorDB_Postman_Collection.json` into Postman
   - The `base_url` variable is set to `http://localhost:8000/v1` (will write in production, change to :8001 if  you want to write in test db)
   - The collection includes examples for all index types (Flat, IVF, LSH SimHash)


## 📚 API Documentation

### Core Endpoints

#### Libraries
- `POST /v1/libraries/` - Create a new library
- `GET /v1/libraries/` - List all libraries
- `GET /v1/libraries/{library_id}` - Get library details
- `PATCH /v1/libraries/{library_id}` - Update library
- `DELETE /v1/libraries/{library_id}` - Delete a library

#### Documents
- `POST /v1/libraries/{library_id}/documents` - Create a document
- `GET /v1/libraries/{library_id}/documents` - List documents in library
- `GET /v1/libraries/{library_id}/documents/{document_id}` - Get document details
- `PATCH /v1/libraries/{library_id}/documents/{document_id}` - Update document
- `DELETE /v1/libraries/{library_id}/documents/{document_id}` - Delete a document

#### Chunks
- `POST /v1/libraries/{library_id}/chunks` - Create a chunk
- `GET /v1/libraries/{library_id}/chunks` - List chunks in library
- `DELETE /v1/libraries/{library_id}/chunks` - Delete all chunks in library
- `POST /v1/libraries/{library_id}/chunks/batch` - Create multiple chunks
- `GET /v1/libraries/{library_id}/chunks/{chunk_id}` - Get chunk details
- `PATCH /v1/libraries/{library_id}/chunks/{chunk_id}` - Update a chunk
- `DELETE /v1/libraries/{library_id}/chunks/{chunk_id}` - Delete a chunk

#### Search & Operations
- `POST /v1/libraries/{library_id}/search` - Vector similarity search
- `POST /v1/libraries/{library_id}/search_text` - Text-based similarity search (auto-generates embedding)
- `POST /v1/libraries/{library_id}/index/train` - Train IVF index
- `POST /v1/libraries/{library_id}/index/rebuild` - Rebuild index
- `GET /v1/libraries/{library_id}/stats` - Get library statistics
- `POST /v1/embed` - Generate embeddings from text

### Example Usage

#### Creating a Library with Flat Index
```bash
curl -X POST "http://localhost:8000/v1/libraries/" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Vector Library",
    "dims": 1024,
    "index_type": "flat",
    "metadata": {"description": "Test library"}
  }'
```

#### Adding a Document
```bash
curl -X POST "http://localhost:8000/v1/libraries/{library_id}/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Sample Document",
    "metadata": {"author": "John Doe"}
  }'
```

#### Creating a Chunk with pre-computed Embedding

The `text` parameter stores the original text content for reference and retrieval purposes

```bash
curl -X POST "http://localhost:8000/v1/libraries/{library_id}/chunks" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "{document_id}",
    "text": "This is sample text for vector embedding.",
    "embedding": [0.1, 0.2, 0.3, ...],
    "metadata": {"chunk_type": "paragraph"}
  }'
```

#### Creating a Chunk with auto-generated embedding

If the `embedding` field is NOT provided, embedding will be auto-generated from the `text` parameter. Then, the `text` parameter is stored for reference and retrieval purposes, as usual.


```bash
curl -X POST "http://localhost:8000/v1/libraries/{library_id}/chunks" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "{document_id}",
    "text": "This is sample text for vector embedding.",
    "embedding": [0.1, 0.2, 0.3, ...],
    "metadata": {"chunk_type": "paragraph"}
  }'
```

#### Vector Search
```bash
curl -X POST "http://localhost:8000/v1/libraries/{library_id}/search" \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [0.1, 0.2, 0.3, ...],
    "k": 5
  }'
```

#### Text-based Search
```bash
curl -X POST "http://localhost:8000/v1/libraries/{library_id}/search_text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "vector database testing",
    "k": 5,
    "include_chunk": true
  }'
```

## 🧠 Indexing Algorithms (and performance)

### 1. Flat Index
- **Type**: Exact search
- **Space Complexity**: O(n*d) where d is the embedding dimension
- **Query Complexity**: O(n*d) where d is the embedding dimension
- **Use Case**: Small datasets, exact similarity search
- **Implementation**: Linear scan through all vectors, return top k most similars

### 2. IVF (Inverted File) Index
- **Type**: Approximate search
- **Space Complexity**: O(n_clusters*d + n*d)
- **Query Complexity**: O(n_clusters*d + C*d)
- **Use Case**: Large datasets, approximate search
- **Implementation**: K-means clustering with inverted lists

### 3. LSH SimHash Index
- **Type**: Approximate search
- **Space Complexity**: O(n_tables*n + n_bits*d)
- **Query Complexity**: O(n_tables*n_bits*d + C*d)
- **Use Case**: High-dimensional vectors, approximate search
- **Implementation**: Random hyperplanes with hash tables

### Algorithm Selection Guide

| Dataset Size | Dimensions | Accuracy Requirement | Recommended Index |
|--------------|------------|---------------------|-------------------|
| Small (< 10K) | Any | Exact | Flat |
| Large (> 100K) | Medium | Approximate | IVF |
| Any | High (> 1000) | Approximate | LSH SimHash |

## 🧪 Testing

The project includes a test suite with **isolated test environments** to ensure tests don't interfere with production data. The test setup uses a dedicated test API service that runs on port 8001 with `TEST_MODE=true`, ensuring all test data goes to the `test` database.

The project includes a data generator (`tests-mongo/data_generator.py`) that creates test data for vector database operations. This generator produces sample text chunks with corresponding embeddings using the Cohere API and simulates CRUD operations (Create, Read, Update, Delete) on libraries, documents, and chunks.

### Running Tests

<!-- TODO: create env!!! for testing and you must have serrvices up -->
#### E2E Tests (uses mongo for persistence)
```bash
cd tests-mongo
# Create venv if not already
python -m venv venv

# Activate venv and install test dependencies if not already)
source venv/bin/activate
pip install -r requirements.txt

# Run tests
```bash
# Automated test runner: runs docker service for the test API
cd tests-mongo
source venv/bin/activate
python run_tests_with_api.py

```

#### Individual Test Suites
```bash
# Flat index tests
python -m pytest test_e2e_mongo.py -v

# IVF index tests  
python -m pytest test_ivf_e2e.py -v

# LSH SimHash tests
python -m pytest test_lsh_simhash_e2e.py -v

# Persistence tests
python -m pytest test_persistence.py -v
```

### Test Environment Isolation

- **Test API**: Runs on port 8001 (separate from production port 8000)
- **Test Database**: Uses `test` database (separate from production `vector_db`)
- **Test Mode**: API runs with `TEST_MODE=true` environment variable
- **Automatic Cleanup**: Test data is automatically cleaned after each test

<!-- #### Performance Tests
```bash
# Run performance comparison
python -m pytest test_lsh_simhash_e2e.py::TestLSHSimHashE2E::test_lsh_simhash_performance_comparison -v -s
``` -->


## ⚙️ Technical Design Choices

### 1. Layered Architecture
**Choice**: Clean separation between API, Vector DB Service, Storage, and Index layers.

**Reason**:
- Easy to test and maintain
- Easy to  switch storage backends and/or implement more sophisticated sotrage orchestrators
- Easy to create new index types

### 2. Thread Safety with AsyncRWLock
**Choice**: Implemented custom AsyncRWLock for concurrent access control.

**Reason**:
- Prevents data races between reads and writes
- Allows multiple concurrent readers
- Ensures exclusive access for writers to avoid starvation
- Critical for production environments

### 3. MongoDB Persistence
**Choice**: Used MongoDB for data persistence.

**Reason**:
- Flexible schema for metadata storage
- Good performance for document-based data
- Ensures data survival across restarts, and can recreate indexes from it
- Indexes are also automatically rebuilt from persistent data when the service starts up

### 4. Pydantic Models
**Choice**: Used Pydantic for data validation and serialization.

**Reason**:
- Automatic API documentation generation
- Runtime type checking and validation
- Clean, (semi) typed declarative schema definition

### Environment Variables

#### Production
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_VERSION=0.1.0

# MongoDB Configuration
MONGODB_DB=vector_db
MONGODB_USER=admin
MONGODB_PASS=password

# Cohere API Configuration (Required)
COHERE_API_KEY=your_cohere_api_key_here
COHERE_EMBED_URL=https://api.cohere.ai/v1/embed
COHERE_MODEL=embed-english-v3.0
```

#### Testing specific
```bash
# Test API Configuration
TEST_BASE_URL=http://localhost:8001/v1  # Test API endpoint
# Uses "test" db for persistency
```

