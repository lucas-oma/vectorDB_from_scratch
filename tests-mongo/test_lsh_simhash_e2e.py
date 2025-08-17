"""
End-to-end tests for LSH SimHash index implementation
Tests the complete workflow using the LSH SimHash index
"""

import os
import time
import pytest
import httpx
from data_generator import generate_test_chunks, generate_embedding

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

class TestLSHSimHashE2E:
    """End-to-end test for LSH SimHash index functionality using API."""
    
    @pytest.fixture
    async def client(self):
        """Create HTTP client for testing."""
        async with httpx.AsyncClient() as client:
            yield client

    @pytest.mark.asyncio
    async def test_lsh_simhash_library_creation(self, client):
        """Test creating a library with LSH SimHash index."""
        create_data = {
            "name": "LSH SimHash Test Library",
            "dims": 1024,
            "index_type": "lsh_simhash",
            "metadata": {"description": "Test library for LSH SimHash index"}
        }
        
        resp = await client.post(_url("/libraries/"), json=create_data)
        _assert_status(resp, 201, "create LSH SimHash library")
        
        library_data = resp.json()
        assert library_data["name"] == create_data["name"]
        assert library_data["dims"] == create_data["dims"]
        assert library_data["index_type"] == create_data["index_type"]
        assert library_data["metadata"] == create_data["metadata"]
        
        print("✅ LSH SimHash library creation test passed!")

    @pytest.mark.asyncio
    async def test_lsh_simhash_basic_workflow(self, client):
        """Test complete workflow: create library, add documents/chunks, search."""
        # Step 1: Create library
        create_data = {
            "name": "LSH SimHash Workflow Test",
            "dims": 1024,
            "index_type": "lsh_simhash",
            "metadata": {"description": "Test LSH SimHash workflow"}
        }
        
        resp = await client.post(_url("/libraries/"), json=create_data)
        _assert_status(resp, 201, "create library")
        
        library_data = resp.json()
        library_id = library_data["id"]
        
        # Step 2: Create document
        document_data = {"title": "LSH SimHash Test Document", "metadata": {}}
        resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document_data)
        _assert_status(resp, 201, "create document")
        
        document_id = resp.json()["id"]
        
        # Step 3: Add chunks with real embeddings
        test_chunks = generate_test_chunks(library_id, document_id, num_chunks=5)
        
        chunk_ids = []
        for chunk in test_chunks:
            chunk_data = {
                "document_id": document_id,
                "text": chunk["text"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"]
            }
            
            resp = await client.post(_url(f"/libraries/{library_id}/chunks"), json=chunk_data)
            _assert_status(resp, 201, "create chunk")
            chunk_ids.append(resp.json()["id"])
        
        # Step 4: Verify chunks were added
        resp = await client.get(_url(f"/libraries/{library_id}/chunks"))
        _assert_status(resp, 200, "get chunks")
        
        chunks = resp.json()
        assert len(chunks) == 5
        
        # Step 5: Test search
        search_embedding = generate_embedding("LSH SimHash vector search test")
        search_data = {
            "embedding": search_embedding,
            "k": 3
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
        _assert_status(resp, 200, "search")
        
        search_results = resp.json()
        assert len(search_results["results"]) > 0
        assert len(search_results["results"]) <= 3
        
        # Verify result structure
        for result in search_results["results"]:
            assert "chunk_id" in result
            assert "similarity_score" in result
            assert isinstance(result["similarity_score"], float)
        
        print("✅ LSH SimHash basic workflow test passed!")

    @pytest.mark.asyncio
    async def test_lsh_simhash_performance_comparison(self, client):
        """Test LSH SimHash vs Flat index performance comparison."""
        # Create LSH SimHash library
        lsh_data = {
            "name": "LSH SimHash Performance Test",
            "dims": 1024,
            "index_type": "lsh_simhash",
            "metadata": {"description": "LSH SimHash performance test"}
        }
        
        resp = await client.post(_url("/libraries/"), json=lsh_data)
        _assert_status(resp, 201, "create LSH SimHash library")
        
        lsh_library_id = resp.json()["id"]
        
        # Create Flat library for comparison
        flat_data = {
            "name": "Flat Performance Test",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {"description": "Flat index performance test"}
        }
        
        resp = await client.post(_url("/libraries/"), json=flat_data)
        _assert_status(resp, 201, "create flat library")
        
        flat_library_id = resp.json()["id"]
        
        # Add same data to both libraries
        for library_id in [lsh_library_id, flat_library_id]:
            # Create document
            document_data = {"title": "Performance Test Document", "metadata": {}}
            resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document_data)
            _assert_status(resp, 201, "create document")
            
            document_id = resp.json()["id"]
            
            # Add chunks
            test_chunks = generate_test_chunks(library_id, document_id, num_chunks=10)
            
            for chunk in test_chunks:
                chunk_data = {
                    "document_id": document_id,
                    "text": chunk["text"],
                    "embedding": chunk["embedding"],
                    "metadata": chunk["metadata"]
                }
                
                resp = await client.post(_url(f"/libraries/{library_id}/chunks"), json=chunk_data)
                _assert_status(resp, 201, "create chunk")
        
        # Test search performance
        search_embedding = generate_embedding("LSH SimHash performance comparison")
        search_data = {
            "embedding": search_embedding,
            "k": 5
        }
        
        # LSH SimHash search
        start_time = time.time()
        resp = await client.post(_url(f"/libraries/{lsh_library_id}/search"), json=search_data)
        _assert_status(resp, 200, "LSH SimHash search")
        lsh_time = time.time() - start_time
        
        lsh_results = resp.json()
        
        # Flat search
        start_time = time.time()
        resp = await client.post(_url(f"/libraries/{flat_library_id}/search"), json=search_data)
        _assert_status(resp, 200, "flat search")
        flat_time = time.time() - start_time
        
        flat_results = resp.json()
        
        # Verify both return results
        assert len(lsh_results["results"]) > 0, "LSH SimHash should return results"
        assert len(flat_results["results"]) > 0, "Flat index should return results"
        
        print(f"✅ LSH SimHash search time: {lsh_time:.4f}s")
        print(f"✅ Flat search time: {flat_time:.4f}s")
        print(f"✅ LSH SimHash returned {len(lsh_results['results'])} results")
        print(f"✅ Flat returned {len(flat_results['results'])} results")
        
        # Note: LSH SimHash is approximate, so results may differ from flat
        # The important thing is that both return results
        print("✅ LSH SimHash performance comparison test passed!")

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 