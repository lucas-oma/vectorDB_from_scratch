"""
End-to-end tests for MongoDB implementation
"""
import os
import time
import pytest
import pytest_asyncio
import httpx
from data_generator import generate_test_data

# Test configuration
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
TEST_MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://admin:password@localhost:27017/vector_db?authSource=admin")
TEST_MONGODB_DB = os.getenv("MONGODB_DB", "vector_db")

def _url(path: str) -> str:
    return f"{BASE_URL}{path}"

def _assert_status(resp: httpx.Response, expected: int, msg: str = ""):
    if resp.status_code != expected:
        raise AssertionError(
            f"{msg} Expected {expected}, got {resp.status_code}. Body: {resp.text}"
        )

class TestMongoE2E:
    """End-to-end tests for MongoDB implementation."""

    @pytest_asyncio.fixture
    async def client(self):
        """Create HTTP client for testing."""
        async with httpx.AsyncClient() as client:
            yield client

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """Test health check endpoint."""
        resp = await client.get(_url("/healthz"))
        _assert_status(resp, 200, "health check")
        
        data = resp.json()
        assert "status" in data
        assert data["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_test_endpoint(self, client):
        """Test the test endpoint to verify MongoDB connection."""
        resp = await client.get(_url("/test"))
        _assert_status(resp, 200, "test endpoint")
        
        data = resp.json()
        assert "mongodb_uri" in data
        assert "mongodb_db" in data
        assert "message" in data
        assert data["message"] == "OK"

    @pytest.mark.asyncio
    async def test_library_crud(self, client):
        """Test complete CRUD operations for libraries."""
        # Create library
        create_data = {
            "name": "Test Library",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {"description": "Test library for E2E tests"}
        }
        
        resp = await client.post(_url("/libraries/"), json=create_data)
        _assert_status(resp, 201, "create library")
        
        library_data = resp.json()
        library_id = library_data["id"]
        
        assert library_data["name"] == create_data["name"]
        assert library_data["dims"] == create_data["dims"]
        assert library_data["index_type"] == create_data["index_type"]
        assert library_data["metadata"] == create_data["metadata"]
        
        # Get library
        resp = await client.get(_url(f"/libraries/{library_id}"))
        _assert_status(resp, 200, "get library")
        
        retrieved_data = resp.json()
        assert retrieved_data["id"] == library_id
        assert retrieved_data["name"] == create_data["name"]
        
        # Update library
        update_data = {
            "name": "Updated Test Library",
            "metadata": {"description": "Updated test library"}
        }
        
        resp = await client.patch(_url(f"/libraries/{library_id}"), json=update_data)
        _assert_status(resp, 200, "update library")
        
        updated_data = resp.json()
        assert updated_data["name"] == update_data["name"]
        assert updated_data["metadata"] == update_data["metadata"]
        
        # List libraries
        resp = await client.get(_url("/libraries/"))
        _assert_status(resp, 200, "list libraries")
        
        libraries = resp.json()
        assert isinstance(libraries, list)
        assert any(lib["id"] == library_id for lib in libraries)
        
        # Delete library
        resp = await client.delete(_url(f"/libraries/{library_id}"))
        _assert_status(resp, 204, "delete library")
        
        # Verify library is deleted
        resp = await client.get(_url(f"/libraries/{library_id}"))
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_library_index_type_change(self, client):
        """Test changing library index type and verifying index rebuild."""
        # Create library with flat index
        create_data = {
            "name": "Index Type Test Library",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {"description": "Test library for index type changes"}
        }
        
        resp = await client.post(_url("/libraries/"), json=create_data)
        _assert_status(resp, 201, "create library")
        
        library_data = resp.json()
        library_id = library_data["id"]
        
        assert library_data["index_type"] == "flat"
        
        # Add some chunks to the library
        document_data = {"title": "Test Document", "metadata": {}}
        resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document_data)
        _assert_status(resp, 201, "create document")
        document_id = resp.json()["id"]
        
        # Create a few chunks
        chunk_data = {
            "document_id": document_id,
            "text": "Test chunk content",
            "embedding": [0.1] * 1024,
            "metadata": {"test": "data"}
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/chunks"), json=chunk_data)
        _assert_status(resp, 201, "create chunk")
        
        # Verify search works with flat index
        search_data = {
            "embedding": [0.1] * 1024,
            "k": 5
        }
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
        _assert_status(resp, 200, "search with flat index")
        
        # Change index type (even though we only have flat for now, this tests the API)
        update_data = {
            "index_type": "flat"  # Keep same type for now, but tests the update path
        }
        
        resp = await client.patch(_url(f"/libraries/{library_id}"), json=update_data)
        _assert_status(resp, 200, "update library index type")
        
        updated_data = resp.json()
        assert updated_data["index_type"] == update_data["index_type"]
        
        # Verify search still works after index type change
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
        _assert_status(resp, 200, "search after index type change")
        
        # Clean up
        resp = await client.delete(_url(f"/libraries/{library_id}"))
        _assert_status(resp, 204, "delete library")

    @pytest.mark.asyncio
    async def test_document_crud(self, client):
        """Test complete CRUD operations for documents."""
        # Create library first
        library_data = {
            "name": "Test Library",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {}
        }
        
        resp = await client.post(_url("/libraries/"), json=library_data)
        _assert_status(resp, 201, "create library")
        library_id = resp.json()["id"]
        
        # Create document
        create_data = {
            "title": "Test Document",
            "metadata": {"author": "Test Author"}
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=create_data)
        _assert_status(resp, 201, "create document")
        
        document_data = resp.json()
        document_id = document_data["id"]
        
        assert document_data["title"] == create_data["title"]
        assert document_data["library_id"] == library_id
        assert document_data["metadata"] == create_data["metadata"]
        
        # Get document
        resp = await client.get(_url(f"/libraries/{library_id}/documents/{document_id}"))
        _assert_status(resp, 200, "get document")
        
        retrieved_data = resp.json()
        assert retrieved_data["id"] == document_id
        assert retrieved_data["title"] == create_data["title"]
        
        # Update document
        update_data = {
            "title": "Updated Test Document",
            "metadata": {"author": "Updated Author"}
        }
        
        resp = await client.patch(_url(f"/libraries/{library_id}/documents/{document_id}"), json=update_data)
        _assert_status(resp, 200, "update document")
        
        updated_data = resp.json()
        assert updated_data["title"] == update_data["title"]
        assert updated_data["metadata"] == update_data["metadata"]
        
        # List documents
        resp = await client.get(_url(f"/libraries/{library_id}/documents"))
        _assert_status(resp, 200, "list documents")
        
        documents = resp.json()
        assert isinstance(documents, list)
        assert any(doc["id"] == document_id for doc in documents)
        
        # Delete document
        resp = await client.delete(_url(f"/libraries/{library_id}/documents/{document_id}"))
        _assert_status(resp, 204, "delete document")
        
        # Verify document is deleted
        resp = await client.get(_url(f"/libraries/{library_id}/documents/{document_id}"))
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_chunk_crud(self, client):
        """Test complete CRUD operations for chunks."""
        # Create library and document first
        library_data = {
            "name": "Test Library",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {}
        }
        
        resp = await client.post(_url("/libraries/"), json=library_data)
        _assert_status(resp, 201, "create library")
        library_id = resp.json()["id"]
        
        document_data = {
            "title": "Test Document",
            "metadata": {}
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document_data)
        _assert_status(resp, 201, "create document")
        document_id = resp.json()["id"]
        
        # Create chunk
        create_data = {
            "document_id": document_id,
            "text": "This is a test chunk for vector search.",
            "embedding": [0.1] * 1024,
            "metadata": {"section": "introduction"}
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/chunks"), json=create_data)
        _assert_status(resp, 201, "create chunk")
        
        chunk_data = resp.json()
        chunk_id = chunk_data["id"]
        
        assert chunk_data["text"] == create_data["text"]
        assert chunk_data["library_id"] == library_id
        assert chunk_data["document_id"] == document_id
        assert chunk_data["embedding"] == create_data["embedding"]
        assert chunk_data["metadata"] == create_data["metadata"]
        
        # Get chunk
        resp = await client.get(_url(f"/libraries/{library_id}/chunks/{chunk_id}"))
        _assert_status(resp, 200, "get chunk")
        
        retrieved_data = resp.json()
        assert retrieved_data["id"] == chunk_id
        assert retrieved_data["text"] == create_data["text"]
        
        # Update chunk
        update_data = {
            "text": "Updated test chunk text.",
            "embedding": [0.2] * 1024,
            "metadata": {"section": "updated"}
        }
        
        resp = await client.patch(_url(f"/libraries/{library_id}/chunks/{chunk_id}"), json=update_data)
        _assert_status(resp, 200, "update chunk")
        
        updated_data = resp.json()
        assert updated_data["text"] == update_data["text"]
        assert updated_data["embedding"] == update_data["embedding"]
        assert updated_data["metadata"] == update_data["metadata"]
        
        # List chunks
        resp = await client.get(_url(f"/libraries/{library_id}/chunks"))
        _assert_status(resp, 200, "list chunks")
        
        chunks = resp.json()
        assert isinstance(chunks, list)
        assert any(chunk["id"] == chunk_id for chunk in chunks)
        
        # Delete chunk
        resp = await client.delete(_url(f"/libraries/{library_id}/chunks/{chunk_id}"))
        _assert_status(resp, 204, "delete chunk")
        
        # Verify chunk is deleted
        resp = await client.get(_url(f"/libraries/{library_id}/chunks/{chunk_id}"))
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_batch_chunk_creation(self, client):
        """Test batch chunk creation."""
        # Create library and document first
        library_data = {
            "name": "Test Library",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {}
        }
        
        resp = await client.post(_url("/libraries/"), json=library_data)
        _assert_status(resp, 201, "create library")
        library_id = resp.json()["id"]
        
        document_data = {
            "title": "Test Document",
            "metadata": {}
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document_data)
        _assert_status(resp, 201, "create document")
        document_id = resp.json()["id"]
        
        # Create chunks in batch
        chunks_data = {
            "chunks": [
                {
                    "document_id": document_id,
                    "text": f"Chunk {i}",
                    "embedding": [0.1 + i * 0.1] * 1024,
                    "metadata": {"index": str(i)}
                }
                for i in range(3)
            ]
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/chunks/batch"), json=chunks_data)
        _assert_status(resp, 201, "create chunks batch")
        
        batch_result = resp.json()
        assert "chunk_ids" in batch_result
        assert len(batch_result["chunk_ids"]) == 3
        
        # Verify chunks were created
        resp = await client.get(_url(f"/libraries/{library_id}/chunks"))
        _assert_status(resp, 200, "list chunks after batch")
        
        chunks = resp.json()
        assert len(chunks) == 3

    @pytest.mark.asyncio
    async def test_vector_search(self, client):
        """Test vector search functionality."""
        # Create library and document
        library_data = {
            "name": "Test Library",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {}
        }
        
        resp = await client.post(_url("/libraries/"), json=library_data)
        _assert_status(resp, 201, "create library")
        library_id = resp.json()["id"]
        
        document_data = {
            "title": "Test Document",
            "metadata": {}
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document_data)
        _assert_status(resp, 201, "create document")
        document_id = resp.json()["id"]
        
        # Create chunks with different embeddings
        embeddings = [
            [0.1] * 1024,  # Similar to query
            [0.2] * 1024,  # Less similar
            [0.9] * 1024,  # Very different
        ]
        
        for i, emb in enumerate(embeddings):
            chunk_data = {
                "document_id": document_id,
                "text": f"Chunk {i}",
                "embedding": emb,
                "metadata": {"index": str(i)}
            }
            
            resp = await client.post(_url(f"/libraries/{library_id}/chunks"), json=chunk_data)
            _assert_status(resp, 201, f"create chunk {i}")
        
        # Rebuild index
        rebuild_data = {"index_type": "flat"}
        resp = await client.post(_url(f"/libraries/{library_id}/index/rebuild"), json=rebuild_data)
        _assert_status(resp, 202, "rebuild index")
        
        # Wait a bit for index to be ready
        time.sleep(0.5)
        
        # Search without chunk details
        search_data = {
            "embedding": [0.15] * 1024,  # Similar to first chunk
            "k": 2,
            "filters": None,
            "include_chunk": False
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
        _assert_status(resp, 200, "search without chunks")
        
        search_result = resp.json()
        assert "library_id" in search_result
        assert "results" in search_result
        assert len(search_result["results"]) == 2
        
        # Search with chunk details
        search_data["include_chunk"] = True
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
        _assert_status(resp, 200, "search with chunks")
        
        search_result = resp.json()
        assert len(search_result["results"]) == 2
        assert all("chunk" in result for result in search_result["results"])

    @pytest.mark.asyncio
    async def test_embedding_endpoint(self, client):
        """Test the embedding endpoint."""
        text_data = {
            "texts": ["This is a test text for embedding."]
        }
        
        resp = await client.post(_url("/embed"), json=text_data)
        _assert_status(resp, 200, "embed text")
        
        embed_result = resp.json()
        assert "embeddings" in embed_result
        assert isinstance(embed_result["embeddings"], list)
        assert len(embed_result["embeddings"]) > 0
        assert len(embed_result["embeddings"][0]) == 1024  # Check dimension

    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test error handling for invalid requests."""
        # Test creating library with invalid data
        invalid_data = {
            "name": "",  # Empty name
            "dims": -1,  # Invalid dimensions
            "index_type": "invalid_type",  # Invalid index type
            "metadata": {}
        }
        
        resp = await client.post(_url("/libraries/"), json=invalid_data)
        assert resp.status_code in [400, 422]  # Validation error
        
        # Test accessing nonexistent resources
        resp = await client.get(_url("/libraries/nonexistent"))
        assert resp.status_code == 404
        
        resp = await client.get(_url("/libraries/nonexistent/documents/nonexistent"))
        assert resp.status_code == 404
        
        # Test invalid search query
        library_data = {
            "name": "Test Library",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {}
        }
        
        resp = await client.post(_url("/libraries/"), json=library_data)
        _assert_status(resp, 201, "create library")
        library_id = resp.json()["id"]
        
        invalid_search = {
            "embedding": [0.1] * 256,  # Wrong dimension
            "k": 2,
            "filters": None,
            "include_chunk": False
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=invalid_search)
        assert resp.status_code in [400, 422]  # Validation error

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, client):
        """Test concurrent operations on the same library."""
        import asyncio
        
        # Create library
        library_data = {
            "name": "Test Library",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {}
        }
        
        resp = await client.post(_url("/libraries/"), json=library_data)
        _assert_status(resp, 201, "create library")
        library_id = resp.json()["id"]
        
        document_data = {
            "title": "Test Document",
            "metadata": {}
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document_data)
        _assert_status(resp, 201, "create document")
        document_id = resp.json()["id"]
        
        # Create chunks concurrently
        async def create_chunk(i):
            chunk_data = {
                "document_id": document_id,
                "text": f"Concurrent chunk {i}",
                "embedding": [0.1 + i * 0.1] * 1024,
                "metadata": {"index": str(i)}
            }
            resp = await client.post(_url(f"/libraries/{library_id}/chunks"), json=chunk_data)
            return resp
        
        tasks = [create_chunk(i) for i in range(5)]
        responses = await asyncio.gather(*tasks)
        
        # Verify all chunks were created successfully
        for resp in responses:
            _assert_status(resp, 201, "concurrent chunk creation")
        
        # Verify total number of chunks
        resp = await client.get(_url(f"/libraries/{library_id}/chunks"))
        _assert_status(resp, 200, "list chunks after concurrent creation")
        
        chunks = resp.json()
        assert len(chunks) == 5

    @pytest.mark.asyncio
    async def test_complete_fixture_workflow(self, client):
        """Test complete workflow using generated fixture data (like original e2e_test.py)."""
        # Generate test data with real embeddings
        test_data = generate_test_data()
        
        # Create libraries
        library_map = {}
        for library in test_data["libraries"]:
            resp = await client.post(_url("/libraries/"), json=library)
            _assert_status(resp, 201, f"create library {library['name']}")
            library_map[library["id"]] = resp.json()["id"]
        
        # Create documents
        document_map = {}
        for document in test_data["documents"]:
            library_id = library_map[document["library_id"]]
            resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document)
            _assert_status(resp, 201, f"create document {document['title']}")
            document_map[document["id"]] = resp.json()["id"]
        
        # Create chunks in batches per library
        chunk_map = {}
        for library in test_data["libraries"]:
            library_id = library_map[library["id"]]
            library_chunks = [chunk for chunk in test_data["chunks"] if chunk["library_id"] == library["id"]]
            
            chunks_data = {
                "chunks": [
                    {
                        "document_id": document_map[chunk["document_id"]],
                        "text": chunk["text"],
                        "embedding": chunk["embedding"],
                        "metadata": chunk["metadata"]
                    }
                    for chunk in library_chunks
                ]
            }
            
            resp = await client.post(_url(f"/libraries/{library_id}/chunks/batch"), json=chunks_data)
            _assert_status(resp, 201, f"create chunks for {library['name']}")
            chunk_ids = resp.json()["chunk_ids"]
            
            for i, chunk in enumerate(library_chunks):
                chunk_map[chunk["id"]] = chunk_ids[i]
        
        # Rebuild indexes
        for library_id in library_map.values():
            rebuild_data = {"index_type": "flat"}
            resp = await client.post(_url(f"/libraries/{library_id}/index/rebuild"), json=rebuild_data)
            _assert_status(resp, 202, f"rebuild index for {library_id}")
        
        # Wait for indexes to be ready
        time.sleep(1.0)
        
        # Test search on each library
        for library in test_data["libraries"]:
            library_id = library_map[library["id"]]
            library_chunks = [chunk for chunk in test_data["chunks"] if chunk["library_id"] == library["id"]]
            
            if library_chunks:
                # Use first chunk's embedding as query
                query_embedding = library_chunks[0]["embedding"]
                search_data = {
                    "embedding": query_embedding,
                    "k": 3,
                    "filters": None,
                    "include_chunk": True
                }
                
                resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
                _assert_status(resp, 200, f"search in {library['name']}")
                
                search_result = resp.json()
                assert search_result["library_id"] == library_id
                assert len(search_result["results"]) > 0
        
        # Test CRUD operations on a sample library
        sample_library_id = list(library_map.values())[0]
        sample_document_id = list(document_map.values())[0]
        
        # Update document
        update_data = {"title": "Updated Document Title"}
        resp = await client.patch(_url(f"/libraries/{sample_library_id}/documents/{sample_document_id}"), json=update_data)
        _assert_status(resp, 200, "update document")
        
        # Update chunk
        sample_chunk_id = list(chunk_map.values())[0]
        chunk_update_data = {"text": "Updated chunk text"}
        resp = await client.patch(_url(f"/libraries/{sample_library_id}/chunks/{sample_chunk_id}"), json=chunk_update_data)
        _assert_status(resp, 200, "update chunk")
        
        # Delete chunk
        resp = await client.delete(_url(f"/libraries/{sample_library_id}/chunks/{sample_chunk_id}"))
        _assert_status(resp, 204, "delete chunk")
        
        # Verify chunk is deleted
        resp = await client.get(_url(f"/libraries/{sample_library_id}/chunks/{sample_chunk_id}"))
        assert resp.status_code == 404
        
        # Delete library
        resp = await client.delete(_url(f"/libraries/{sample_library_id}"))
        _assert_status(resp, 204, "delete library")
        
        # Verify library is deleted
        resp = await client.get(_url(f"/libraries/{sample_library_id}"))
        assert resp.status_code == 404 