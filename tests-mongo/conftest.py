"""
Pytest configuration for E2E tests
"""
import asyncio
import os
import pytest
import httpx
from typing import AsyncGenerator, Generator

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient, None]:
    """Create an async HTTP client for testing the API."""
    # Use TEST_BASE_URL without /v1 suffix for base_url
    test_base_url = os.getenv("TEST_BASE_URL", "http://localhost:8001/v1").replace("/v1", "")
    async with httpx.AsyncClient(base_url=test_base_url) as client:
        yield client

@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Clean up ALL test data after each test to prevent conflicts."""
    # Create a separate client for cleanup (fixes small bug of loading client before it was ready)
    test_base_url = os.getenv("TEST_BASE_URL", "http://localhost:8001/v1").replace("/v1", "")
    async with httpx.AsyncClient(base_url=test_base_url) as client:
        yield  # Run the test
        
        # Clean up ALL libraries after each test
        try:
            resp = await client.get("/v1/libraries/")
            if resp.status_code == 200:
                libraries = resp.json()
                print(f"Cleaning up {len(libraries)} libraries after test")
                cleaned_count = 0
                for library in libraries:
                    try:
                        delete_resp = await client.delete(f"/v1/libraries/{library['id']}")
                        if delete_resp.status_code == 200:
                            cleaned_count += 1
                            print(f"Cleaned up library: {library['name']}")
                    except Exception as e:
                        print(f"Error deleting library {library['name']}: {e}")
                print(f"Cleaned up {cleaned_count} libraries")
        except Exception as e:
            print(f"Error during cleanup: {e}")


 