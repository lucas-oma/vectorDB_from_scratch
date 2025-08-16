"""
Persistence tests for MongoDB implementation
Tests that data actually persists in MongoDB across container restarts
"""

import os
import time
import pytest
import httpx
import subprocess
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

class TestPersistence:
    """Test that data persists in MongoDB across container restarts."""
    
    @pytest.fixture
    async def client(self):
        """Create HTTP client for testing."""
        async with httpx.AsyncClient() as client:
            yield client

    @pytest.mark.asyncio
    async def test_data_persistence_across_restart(self, client):
        """Test that data persists when containers are restarted."""
        
        # Step 1: Create library
        create_data = {
            "name": "Persistence Test Library",
            "dims": 1024,
            "index_type": "flat",
            "metadata": {"description": "Test library for persistence"}
        }
        
        resp = await client.post(_url("/libraries/"), json=create_data)
        _assert_status(resp, 201, "create library")
        
        library_data = resp.json()
        library_id = library_data["id"]
        
        # Step 2: Create document
        document_data = {"title": "Persistence Test Document", "metadata": {}}
        resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document_data)
        _assert_status(resp, 201, "create document")
        document_id = resp.json()["id"]
        
        # Step 3: Create chunks with real embeddings
        test_chunks = generate_test_chunks(library_id, document_id, num_chunks=3)
        
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
        
        # Step 4: Verify data exists before restart
        resp = await client.get(_url(f"/libraries/{library_id}"))
        _assert_status(resp, 200, "get library before restart")
        
        resp = await client.get(_url(f"/libraries/{library_id}/documents"))
        _assert_status(resp, 200, "get documents before restart")
        documents = resp.json()
        assert len(documents) == 1
        assert documents[0]["id"] == document_id
        
        resp = await client.get(_url(f"/libraries/{library_id}/chunks"))
        _assert_status(resp, 200, "get chunks before restart")
        chunks = resp.json()
        assert len(chunks) == 3
        
        # Step 5: Restart containers
        print("🔄 Restarting containers to test persistence...")
        
        # Get the project directory (parent of tests-mongo)
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Restart containers
        subprocess.run(["docker-compose", "restart"], cwd=project_dir, check=True)
        
        # Wait for containers to be ready
        print("⏳ Waiting for containers to be ready...30s")
        time.sleep(30)
        
        # Wait for health checks to pass
        max_retries = 10
        for i in range(max_retries):
            try:
                resp = await client.get(_url("/healthz"))
                if resp.status_code == 200:
                    print("✅ Containers are healthy")
                    break
            except Exception:
                pass
            
            if i < max_retries - 1:
                print(f"⏳ Waiting for health check... ({i+1}/{max_retries})")
                time.sleep(5)
        else:
            raise Exception("Containers failed to become healthy after restart")
        
        # Step 6: Verify data still exists after restart
        print("🔍 Verifying data persistence...")
        
        # Check library still exists
        resp = await client.get(_url(f"/libraries/{library_id}"))
        _assert_status(resp, 200, "get library after restart")
        
        library_after = resp.json()
        assert library_after["id"] == library_id
        assert library_after["name"] == create_data["name"]
        assert library_after["dims"] == create_data["dims"]
        assert library_after["index_type"] == create_data["index_type"]
        
        # Check document still exists
        resp = await client.get(_url(f"/libraries/{library_id}/documents"))
        _assert_status(resp, 200, "get documents after restart")
        documents_after = resp.json()
        assert len(documents_after) == 1
        assert documents_after[0]["id"] == document_id
        assert documents_after[0]["title"] == document_data["title"]
        
        # Check chunks still exist
        resp = await client.get(_url(f"/libraries/{library_id}/chunks"))
        _assert_status(resp, 200, "get chunks after restart")
        chunks_after = resp.json()
        assert len(chunks_after) == 3
        
        # Verify chunk IDs match
        chunk_ids_after = [chunk["id"] for chunk in chunks_after]
        assert set(chunk_ids_after) == set(chunk_ids)
        
        # Step 7: Test that search works after index rebuild
        # Note: Indexes are in-memory and need to be rebuilt after restart
        # The system should automatically rebuild indexes when needed
        
        # For flat index, it should work automatically
        search_embedding = generate_embedding("MongoDB persistence test")
        search_data = {
            "embedding": search_embedding,
            "k": 3
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
        _assert_status(resp, 200, "search after restart")
        
        search_results = resp.json()
        assert len(search_results["results"]) > 0
        
        print("✅ Data persistence test passed!")

    @pytest.mark.asyncio
    async def test_ivf_persistence_across_restart(self, client):
        """Test that IVF index data persists when containers are restarted."""
        
        # Step 1: Create IVF library
        create_data = {
            "name": "IVF Persistence Test Library",
            "dims": 1024,
            "index_type": "ivf",
            "metadata": {"description": "Test IVF library for persistence"}
        }
        
        resp = await client.post(_url("/libraries/"), json=create_data)
        _assert_status(resp, 201, "create IVF library")
        
        library_data = resp.json()
        library_id = library_data["id"]
        
        # Step 2: Create document and chunks
        document_data = {"title": "IVF Persistence Test Document", "metadata": {}}
        resp = await client.post(_url(f"/libraries/{library_id}/documents"), json=document_data)
        _assert_status(resp, 201, "create document")
        document_id = resp.json()["id"]
        
        # Create training chunks
        test_chunks = generate_test_chunks(library_id, document_id, num_chunks=5)
        
        for chunk in test_chunks:
            chunk_data = {
                "document_id": document_id,
                "text": chunk["text"],
                "embedding": chunk["embedding"],
                "metadata": chunk["metadata"]
            }
            
            resp = await client.post(_url(f"/libraries/{library_id}/chunks"), json=chunk_data)
            _assert_status(resp, 201, "create chunk")
        
        # Step 3: Train the IVF index
        training_vectors = [chunk["embedding"] for chunk in test_chunks]
        train_data = {"sample_vectors": training_vectors}
        
        resp = await client.post(_url(f"/libraries/{library_id}/index/train"), json=train_data)
        _assert_status(resp, 202, "train IVF index")
        
        # Step 4: Test search before restart
        search_embedding = generate_embedding("IVF persistence test")
        search_data = {
            "embedding": search_embedding,
            "k": 3
        }
        
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
        _assert_status(resp, 200, "search before restart")
        
        search_results_before = resp.json()
        assert len(search_results_before["results"]) > 0
        
        # Step 5: Restart containers
        print("🔄 Restarting containers to test IVF persistence...")
        
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        subprocess.run(["docker-compose", "restart"], cwd=project_dir, check=True)
        
        # Wait for containers to be ready
        print("⏳ Waiting for containers to be ready...")
        time.sleep(30)
        
        # Wait for health checks to pass
        max_retries = 10
        for i in range(max_retries):
            try:
                resp = await client.get(_url("/healthz"))
                if resp.status_code == 200:
                    print("✅ Containers are healthy")
                    break
            except Exception:
                pass
            
            if i < max_retries - 1:
                print(f"⏳ Waiting for health check... ({i+1}/{max_retries})")
                time.sleep(5)
        else:
            raise Exception("Containers failed to become healthy after restart")
        
        # Step 6: Verify IVF data still exists and works
        print("🔍 Verifying IVF data persistence...")
        
        # Check library still exists
        resp = await client.get(_url(f"/libraries/{library_id}"))
        _assert_status(resp, 200, "get IVF library after restart")
        
        library_after = resp.json()
        assert library_after["id"] == library_id
        assert library_after["index_type"] == "ivf"
        
        # Check chunks still exist
        resp = await client.get(_url(f"/libraries/{library_id}/chunks"))
        _assert_status(resp, 200, "get chunks after restart")
        chunks_after = resp.json()
        assert len(chunks_after) == 5
        
        # Test that IVF index needs retraining after restart
        # IVF indexes are not persisted and need to be retrained
        search_data = {
            "embedding": search_embedding,
            "k": 3
        }
        
        # This should fail because IVF index is not trained after restart
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
        assert resp.status_code == 500, "IVF search should fail without retraining"
        
        # Retrain the IVF index
        training_vectors = [chunk["embedding"] for chunk in chunks_after]
        train_data = {"sample_vectors": training_vectors}
        
        resp = await client.post(_url(f"/libraries/{library_id}/index/train"), json=train_data)
        _assert_status(resp, 202, "retrain IVF index")
        
        # Now search should work
        resp = await client.post(_url(f"/libraries/{library_id}/search"), json=search_data)
        _assert_status(resp, 200, "search after retraining")
        
        search_results_after = resp.json()
        assert len(search_results_after["results"]) > 0
        
        print("✅ IVF data persistence test passed!") 