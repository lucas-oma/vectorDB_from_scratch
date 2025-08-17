from __future__ import annotations
import asyncio
from typing import List, Optional, Dict, Type
from uuid import uuid4
from collections import defaultdict
from contextlib import asynccontextmanager
import numpy as np

from app.core.indexes.ivf import IVFIndex
from app.core.models import Library, Document, Chunk, SearchResult
from app.core.mongo_storage import MongoStorage
from app.core.indexing import VectorIndex
from app.core.indexes.flat import FlatIndex

"""
=============================================

Per-library async read/write lock

=============================================
"""

class AsyncRWLock:
    def __init__(self):
        self._readers = 0
        self._rlock = asyncio.Lock()
        self._wlock = asyncio.Lock()
        self._turnstile = asyncio.Lock()  # prevents writer starvation

    @asynccontextmanager
    async def read(self):
        # Let waiting writers go first
        async with self._turnstile:
            pass
        async with self._rlock:
            self._readers += 1
            if self._readers == 1:
                await self._wlock.acquire()
        try:
            yield
        finally:
            async with self._rlock:
                self._readers -= 1
                if self._readers == 0:
                    self._wlock.release()

    @asynccontextmanager
    async def write(self):
        await self._turnstile.acquire()
        await self._wlock.acquire()
        try:
            yield
        finally:
            self._wlock.release()
            self._turnstile.release()



"""
=============================================

Vector DB service (Mongo-backed)

=============================================
"""

class VectorDBService:
    """
    Service for managing vector DB operations with multiple index types supported
    - Persistence: MongoStorage
    - In-memory index: VectorIndex implementation per library (flat/hnsw/lsh)
    """

    def __init__(self, storage: MongoStorage, *, default_index_type: str = "flat"):
        self.storage = storage
        self.indexes: Dict[str, VectorIndex] = {} # library_id -> in-memory index instance

        self._index_locks: Dict[str, AsyncRWLock] = defaultdict(AsyncRWLock) # library_id -> per-library lock
        self._service_lock = asyncio.Lock() # internal lock

        self._index_registry: Dict[str, Type[VectorIndex]] = {
            "flat": FlatIndex,
            "ivf": IVFIndex,
        }
        if default_index_type not in self._index_registry:
            raise ValueError(f"default_index_type '{default_index_type}' not in registry")
        self._default_index_type = default_index_type

        # Caching per-library index types to avoid extra reads
        self._lib_index_type: Dict[str, str] = {}

    async def _get_idx_lock(self, lib_id: str) -> AsyncRWLock:
        async with self._service_lock:
            return self._index_locks[lib_id]

    def _resolve_index_cls(self, index_type: Optional[str]) -> Type[VectorIndex]:
        """Resolve index class from index type"""
        t = (index_type or self._default_index_type).lower()
        cls = self._index_registry.get(t)
        if cls is None:
            cls = self._index_registry[self._default_index_type]
        return cls

    # ---------------- libraries ----------------
    async def create_library(self, name: str, dims: int, index_type: str, metadata: dict) -> Library:
        lib = Library(id=str(uuid4()), name=name, dims=dims, index_type=index_type, metadata=metadata or {})
        await self.storage.save_library(lib)

        idx_cls = self._resolve_index_cls(lib.index_type)
        async with self._service_lock:
            self.indexes[lib.id] = idx_cls(dimension=lib.dims)  # create empty index
            _ = self._index_locks[lib.id]                       # ensure per-lib lock exists
            self._lib_index_type[lib.id] = lib.index_type.lower()
        return lib

    async def get_library(self, lib_id: str) -> Optional[Library]:
        return await self.storage.load_library(lib_id)

    async def list_libraries(self) -> List[Library]:
        libs = await self.storage.load_all_libraries()
        if isinstance(libs, dict):
            return [Library(**v) if not isinstance(v, Library) else v for v in libs.values()]
        return list(libs or [])

    async def update_library(self, lib_id: str, updates: dict) -> Optional[Library]:
        # Dims change not supported
        lib = await self.storage.update_library(lib_id, updates)
        if lib:
            # keep cache in sync if index_type changed
            if "index_type" in updates and updates["index_type"] != lib.index_type:
                async with self._service_lock:
                    self._lib_index_type[lib_id] = lib.index_type.lower()
                # Rebuild index with new type
                await self.rebuild_index(lib_id)
        return lib

    async def delete_library(self, lib_id: str) -> bool:
        lock = await self._get_idx_lock(lib_id)
        async with lock.write():
            if not await self.get_library(lib_id):
                return False
            await self.storage.delete_chunks_for_library(lib_id)
            await self.storage.delete_documents_for_library(lib_id)
            ok = await self.storage.delete_library(lib_id)

        async with self._service_lock:
            self.indexes.pop(lib_id, None)
            self._index_locks.pop(lib_id, None)
            self._lib_index_type.pop(lib_id, None)
        return ok

    # ---------------- documents ----------------
    async def create_document(self, lib_id: str, title: str, metadata: dict) -> Document:
        if not await self.get_library(lib_id):
            raise KeyError("library")
        doc = Document(id=str(uuid4()), library_id=lib_id, title=title, metadata=metadata or {})
        await self.storage.save_document(doc)
        return doc

    async def get_document(self, doc_id: str) -> Optional[Document]:
        return await self.storage.load_document(doc_id)

    async def list_documents(self, lib_id: str) -> List[Document]:
        return await self.storage.load_documents_for_library(lib_id) or []

    async def update_document(self, doc_id: str, updates: dict) -> Optional[Document]:
        if "library_id" in updates:
            raise ValueError("Changing document.library_id is not supported")
        return await self.storage.update_document(doc_id, updates)

    async def delete_document(self, lib_id: str, doc_id: str) -> bool:
        doc = await self.get_document(doc_id)
        if not doc or doc.library_id != lib_id:
            return False
        lock = await self._get_idx_lock(lib_id)
        async with lock.write():
            idx = self.indexes.get(lib_id)
            if idx:
                for chunk in await self.storage.load_chunks_for_document(doc_id) or []:
                    idx.remove_chunk(chunk.id)
        await self.storage.delete_chunks_for_document(doc_id)
        return await self.storage.delete_document(doc_id)

    # ---------------- chunks ----------------
    async def create_chunk(self, lib_id: str, doc_id: str, text: str, embedding: List[float], metadata: dict) -> Chunk:
        lib = await self.get_library(lib_id)
        doc = await self.get_document(doc_id)
        if not lib:
            raise KeyError("library")
        if not doc or doc.library_id != lib_id:
            raise KeyError("document")
        if len(embedding) != lib.dims:
            raise ValueError(f"dim mismatch: {len(embedding)} != {lib.dims}")

        ch = Chunk(
            id=str(uuid4()),
            library_id=lib_id,
            document_id=doc_id,
            text=text,
            embedding=embedding,
            metadata=metadata or {},
        )
        await self.storage.save_chunk(ch)

        lock = await self._get_idx_lock(lib_id)
        async with lock.write():
            idx = self._ensure_index_sync(lib_id, lib.dims, lib.index_type)
            idx.add_chunk(ch)
        return ch

    async def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return await self.storage.load_chunk(chunk_id)

    async def list_chunks(self, lib_id: str) -> List[Chunk]:
        lib = await self.get_library(lib_id)
        if not lib:
            raise KeyError("library")
        return await self.storage.load_chunks_for_library(lib_id) or []

    async def update_chunk(self, lib_id: str, chunk_id: str, **updates) -> Optional[Chunk]:
        ch = await self.storage.load_chunk(chunk_id)
        if not ch or ch.library_id != lib_id:
            return None
        if "embedding" in updates:
            lib = await self.get_library(lib_id)
            if not lib or len(updates["embedding"]) != lib.dims:
                raise ValueError(f"dim mismatch: {len(updates['embedding'])} != {lib.dims}")

        updated = await self.storage.update_chunk(chunk_id, updates)
        if updated and "embedding" in updates:
            lock = await self._get_idx_lock(lib_id)
            async with lock.write():
                lib = await self.get_library(lib_id) # use the lib's index type
                idx = self._ensure_index_sync(lib_id, lib.dims, lib.index_type)
                idx.update_chunk(chunk_id, updated)
        return updated

    async def delete_chunk(self, lib_id: str, chunk_id: str) -> bool:
        ok = await self.storage.delete_chunk(chunk_id)
        if ok:
            lock = await self._get_idx_lock(lib_id)
            async with lock.write():
                idx = self.indexes.get(lib_id)
                if idx:
                    idx.remove_chunk(chunk_id)
        return ok

    # ---------------- search and index operations ----------------
    async def search(self, lib_id: str, query: List[float], k: int, include_chunk: bool = False) -> List[SearchResult]:
        """
        Search the index for the k most similar chunks to the query vector. 
        If include_chunk is True, the chunk object is returned.
        """
        lib = await self.get_library(lib_id)
        if not lib:
            raise KeyError("library")
        if len(query) != lib.dims:
            raise ValueError(f"dim mismatch: {len(query)} != {lib.dims}")

        idx = await self._ensure_index(lib_id, lib.dims, lib.index_type)

        lock = await self._get_idx_lock(lib_id)
        async with lock.read():
            idx = self.indexes.get(lib_id) or idx
            results = idx.search(query_embedding=query, k=k)

        if include_chunk:
            for r in results:
                r.chunk = await self.storage.load_chunk(r.chunk_id)
        return results

    async def rebuild_index(self, lib_id: str) -> None:
        lib = await self.get_library(lib_id)
        if not lib:
            raise KeyError("library")

        # Validate that the library's index type is supported
        if lib.index_type.lower() not in self._index_registry:
            supported_types = list(self._index_registry.keys())
            raise ValueError(f"Index type '{lib.index_type}' not supported. Supported types: {supported_types}")

        # Build to the side
        idx_cls = self._resolve_index_cls(lib.index_type)
        new_idx: VectorIndex = idx_cls(dimension=lib.dims)
        for c in await self.storage.load_chunks_for_library(lib_id) or []:
            new_idx.add_chunk(c)

        # Update index (with lock)
        lock = await self._get_idx_lock(lib_id)
        async with lock.write():
            async with self._service_lock:
                self.indexes[lib_id] = new_idx

    async def train_index(self, lib_id: str, sample_vectors: Optional[List[List[float]]] = None) -> None:
        """
        Train an index (e.g., IVF). Uses proper locking to ensure thread safety.
        """
        lib = await self.get_library(lib_id)
        if not lib:
            raise KeyError("library")

        # Validate that the library's index type is supported
        if lib.index_type.lower() not in self._index_registry:
            supported_types = list(self._index_registry.keys())
            raise ValueError(f"Index type '{lib.index_type}' not supported. Supported types: {supported_types}")

        # Get the index with proper locking
        lock = await self._get_idx_lock(lib_id)
        async with lock.write():
            idx = self.indexes.get(lib_id)
            if not idx:
                raise ValueError("Index not found")

            # Train the index
            if sample_vectors:
                sample_vectors_np = np.array(sample_vectors, dtype=np.float32)
                idx.train(sample_vectors=sample_vectors_np)
            else:
                # Use existing vectors in the index
                if hasattr(idx, 'chunk_vectors') and idx.chunk_vectors:
                    sample_vectors_np = np.stack(list(idx.chunk_vectors.values()))
                    idx.train(sample_vectors=sample_vectors_np)
                else:
                    raise ValueError("No vectors available for training")

    # ---------------- index helpers ----------------
    async def _ensure_index(self, lib_id: str, dims: int, index_type: Optional[str]) -> VectorIndex:
        """
        Ensure an index exists for a library. Safe for callers without the per-lib lock.
        May acquire the per-lib WRITE lock if the index is missing.
        Automatically rebuilds index from persisted data if it's missing.
        """
        idx = self.indexes.get(lib_id)
        if idx is not None:
            return idx

        # Per-lib write lock
        lock = await self._get_idx_lock(lib_id)
        async with lock.write():
            idx = self.indexes.get(lib_id)
            if idx is None:
                # Create new index
                idx_cls = self._resolve_index_cls(index_type or self._lib_index_type.get(lib_id))
                idx = idx_cls(dimension=dims)
                
                # Populate index with existing chunks from MongoDB
                chunks = await self.storage.load_chunks_for_library(lib_id) or []
                for chunk in chunks:
                    idx.add_chunk(chunk)
                
                self.indexes[lib_id] = idx
                if index_type:
                    async with self._service_lock:
                        self._lib_index_type[lib_id] = index_type.lower()
            return idx

    def _ensure_index_sync(self, lib_id: str, dims: int, index_type: Optional[str]) -> VectorIndex:
        """
        Ensure index exists; intended for callers that already hold the WRITE lock!!
        """
        idx = self.indexes.get(lib_id)
        if idx is None:
            idx_cls = self._resolve_index_cls(index_type or self._lib_index_type.get(lib_id))
            idx = idx_cls(dimension=dims)
            self.indexes[lib_id] = idx
            if index_type:
                self._lib_index_type[lib_id] = index_type.lower()
        return idx
