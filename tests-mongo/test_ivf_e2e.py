"""
E2E test for IVF index functionality using API endpoints
Tests the complete workflow: create library -> add data -> train -> search
"""

import os
import time
import pytest
import httpx
import numpy as np
from typing import List
from data_generator import generate_test_chunks, generate_embedding

# Load environment variables from root .env file
try:
    from dotenv import load_dotenv
    load_dotenv("../.env")
except ImportError:
    pass

# Test configuration - Use test API on port 8001
BASE_URL = os.getenv("TEST_BASE_URL", "http://localhost:8001/v1")
# Use credentials from .env file, connect to test database
user = os.getenv("MONGODB_USER", "admin")
password = os.getenv("MONGODB_PASS", "password")
TEST_MONGODB_URI = f"mongodb://{user}:{password}@localhost:27017/test?authSource=admin"
TEST_MONGODB_DB = "test"

def _url(path: str) -> str:
    return f"{BASE_URL}{path}"

def _assert_status(resp: httpx.Response, expected: int, msg: str = ""):
    if resp.status_code != expected:
        raise AssertionError(
            f"{msg} Expected {expected}, got {resp.status_code}. Body: {resp.text}"
        )

class TestIVFE2E:
    """End-to-end test for IVF index functionality using API."""
    

    @pytest.mark.asyncio
    async def test_ivf_library_creation(self, client):
        """Test creating an IVF library via API."""
        create_data = {
            "name": "Test IVF Library",
            "dims": 1024,
            "index_type": "ivf",
            "metadata": {"description": "Test IVF library for E2E tests"}
        }
        
        resp = await client.post(_url("/libraries/"), json=create_data)
        _assert_status(resp, 201, "create IVF library")
        
        library_data = resp.json()
        assert library_data["name"] == create_data["name"]
        assert library_data["dims"] == create_data["dims"]
        assert library_data["index_type"] == "ivf"
        assert library_data["metadata"] == create_data["metadata"]
        
        print("✅ IVF library creation test passed!")
        return library_data["id"]
    
    @pytest.mark.asyncio
    async def test_ivf_basic_workflow(self, client):
        """Test basic IVF workflow: create library -> add data -> train -> search."""
        # Create library with IVF index
        create_data = {
            "name": "IVF Test Library",
            "dims": 1024,
            "index_type": "ivf",
            "metadata": {"description": "Test library for IVF workflow"}
        }
        
        resp = await client.post(_url("/libraries/"), json=create_data)
        _assert_status(resp, 201, "create library")
        
        library_data = resp.json()
        library_id = library_data["id"]
        
        assert library_data["index_type"] == "ivf"
        
                # Add some chunks to the library (these will be stored but not indexed yet)
        document_data = {"title": "Test Document", "metadata": {}}
        resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document_data)
        _assert_status(resp, 201, "create document")
        document_id = resp.json()["id"]

        # Generate training chunks with real embeddings
        training_chunks = generate_test_chunks(library_id, document_id, num_chunks=5)
        
        # Add training chunks to the library
        for i, chunk in enumerate(training_chunks):
            chunk_data = {
                "document_id": document_id,
                "text": chunk["text"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"]
            }

            resp = await client.post(_url(f"/libraries/{library_id}/chunks"), json=chunk_data)
            _assert_status(resp, 201, f"create training chunk {i}")

        # Train the IVF index with sample vectors (this is the key difference from flat index)
        print("Training IVF index...")
        training_vectors = [chunk["embedding"] for chunk in training_chunks]
        train_data = {"sample_vectors": training_vectors}
        resp = await client.post(_url(f"/libraries/{library_id}/index/train"), json=train_data)
        _assert_status(resp, 202, "train IVF index")
        
        # Verify search works with IVF index
        search_embedding = generate_embedding("MongoDB vector database search")
        search_data = {
            "embedding": search_embedding,
            "k": 5
        }
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
        _assert_status(resp, 200, "search with IVF index")
        
        # Clean up
        resp = await client.delete(_url(f"/libraries/{library_id}"))
        _assert_status(resp, 204, "delete library")
        
        print("✅ IVF basic workflow test passed!")
    
    @pytest.mark.asyncio
    async def test_ivf_performance_comparison(self, client):
        """Compare IVF vs Flat index performance using API."""
        
        # Create IVF library
        ivf_data = {
            "name": "Test IVF Performance",
            "dims": 1024,
            "index_type": "ivf",
            "metadata": {"test": "performance"}
        }
        
        resp = await client.post(_url("/libraries/"), json=ivf_data)
        _assert_status(resp, 201, "create IVF library")
        ivf_library_id = resp.json()["id"]
        
        # Create Flat library
        flat_data = {
            "name": "Test Flat Performance",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {"test": "performance"}
        }
        
        resp = await client.post(_url("/libraries/"), json=flat_data)
        _assert_status(resp, 201, "create flat library")
        flat_library_id = resp.json()["id"]
        
        # Create documents for both
        doc_data = {"title": "Performance Test Document", "metadata": {"type": "test"}}
        
        resp = await client.post(_url(f"/libraries/{ivf_library_id}/documents"), json=doc_data)
        _assert_status(resp, 201, "create IVF document")
        ivf_document_id = resp.json()["id"]
        
        resp = await client.post(_url(f"/libraries/{flat_library_id}/documents"), json=doc_data)
        _assert_status(resp, 201, "create flat document")
        flat_document_id = resp.json()["id"]
        
        # Generate training chunks with real embeddings
        training_chunks = generate_test_chunks(ivf_library_id, ivf_document_id, num_chunks=10)
        
        # Add training chunks to IVF
        for i, chunk in enumerate(training_chunks):
            chunk_data = {
                "document_id": ivf_document_id,
                "text": chunk["text"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"]
            }
            
            # Add to IVF
            resp = await client.post(
                _url(f"/libraries/{ivf_library_id}/chunks"),
                json=chunk_data
            )
            _assert_status(resp, 201, f"add training chunk {i} to IVF")
        
        # Train the IVF index with sample vectors
        print("Training IVF index for performance test...")
        training_vectors = [chunk["embedding"] for chunk in training_chunks]
        train_data = {"sample_vectors": training_vectors}
        resp = await client.post(_url(f"/libraries/{ivf_library_id}/index/train"), json=train_data)
        _assert_status(resp, 202, "train IVF index")
        
        # Add training chunks to flat library as well
        for i, chunk in enumerate(training_chunks):
            flat_chunk_data = {
                "document_id": flat_document_id,
                "text": chunk["text"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"]
            }
            resp = await client.post(
                _url(f"/libraries/{flat_library_id}/chunks"),
                json=flat_chunk_data
            )
            _assert_status(resp, 201, f"add training chunk {i} to flat")
        
        # Generate additional chunks for performance testing
        additional_chunks = generate_test_chunks(ivf_library_id, ivf_document_id, num_chunks=90)
        
        # Add remaining chunks to both libraries
        for i, chunk in enumerate(additional_chunks):
            chunk_data = {
                "document_id": ivf_document_id,
                "text": chunk["text"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"]
            }
            
            # Add to IVF
            resp = await client.post(
                _url(f"/libraries/{ivf_library_id}/chunks"),
                json=chunk_data
            )
            _assert_status(resp, 201, f"add chunk {i} to IVF")
            
            # Add to flat
            flat_chunk_data = {
                "document_id": flat_document_id,
                "text": chunk["text"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"]
            }
            resp = await client.post(
                _url(f"/libraries/{flat_library_id}/chunks"),
                json=flat_chunk_data
            )
            _assert_status(resp, 201, f"add chunk {i} to flat")
        

        
        # Test search performance
        query_embedding = generate_embedding("vector database performance test")
        search_data = {"embedding": query_embedding, "k": 10}
        
        # Time IVF search
        start = time.time()
        ivf_response = await client.post(_url(f"/libraries/{ivf_library_id}/search"), json=search_data)
        _assert_status(ivf_response, 200, "IVF search")
        ivf_time = time.time() - start
        
        # Time flat search
        start = time.time()
        flat_response = await client.post(_url(f"/libraries/{flat_library_id}/search"), json=search_data)
        _assert_status(flat_response, 200, "flat search")
        flat_time = time.time() - start
        
        print(f"IVF search time: {ivf_time:.4f}s")
        print(f"Flat search time: {flat_time:.4f}s")
        print(f"Speedup: {flat_time/ivf_time:.2f}x")
        
        # Verify both searches returned results
        ivf_data = ivf_response.json()
        flat_data = flat_response.json()
        assert len(ivf_data["results"]) > 0, "IVF search should return results"
        assert len(flat_data["results"]) > 0, "Flat search should return results"
        
        # Note: IVF and flat may return different top results due to approximate vs exact search
        # This is expected behavior for IVF indexes
        print(f"IVF top result: {ivf_data['results'][0]['chunk_id']}")
        print(f"Flat top result: {flat_data['results'][0]['chunk_id']}")
        
        print("✅ IVF vs Flat performance comparison test passed!")


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 