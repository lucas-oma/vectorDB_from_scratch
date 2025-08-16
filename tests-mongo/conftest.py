"""
Pytest configuration for E2E tests
"""
import asyncio
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
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        yield client 