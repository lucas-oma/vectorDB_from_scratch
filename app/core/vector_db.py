from __future__ import annotations
from typing import List, Optional
from uuid import uuid4

from app.core.models import Library, Document, Chunk, SearchResult
from app.core.storage import DiskStorage
from app.core.indexes.flat import FlatIndex

# TODO: instead of having Flatindex hardcoded, add class param to control index type
class VectorDBService:
    """Service for managing vector database operations. Deals with persisting and indexing operations."""
    
    def __init__(self, storage: DiskStorage):
        self.storage = storage
        self.indexes: dict[str, FlatIndex] = {}  # library_id -> index

    # -------- libraries --------
    def create_library(self, name: str, dims: int, index_type: str, metadata: dict) -> Library:
        lib = Library(id=str(uuid4()), name=name, dims=dims, index_type=index_type, metadata=metadata or {})
        self.storage.save_library(lib)
        self.indexes[lib.id] = FlatIndex(dimension=lib.dims)
        return lib

    def get_library(self, lib_id: str) -> Optional[Library]:
        return self.storage.load_library(lib_id)

    def list_libraries(self) -> List[Library]:
        libs = self.storage.load_all_libraries()
        if isinstance(libs, dict):
            return [Library(**v) if not isinstance(v, Library) else v for v in libs.values()]
        return list(libs or [])

    def update_library(self, lib_id: str, updates: dict) -> Optional[Library]:
        # dims change not supported
        return self.storage.update_library(lib_id, updates)

    def delete_library(self, lib_id: str) -> bool:
        if not self.get_library(lib_id):
            return False
        
        # Remove from memory index
        self.indexes.pop(lib_id, None)
        
        # Delete from disk
        self.storage.delete_chunks_for_library(lib_id)
        self.storage.delete_documents_for_library(lib_id)
        self.storage.delete_snapshot_files(lib_id) # TODO: [Work in progress]
        return self.storage.delete_library(lib_id)

    # -------- documents --------
    def create_document(self, lib_id: str, title: str, metadata: dict) -> Document:
        if not self.get_library(lib_id): raise KeyError("library")
        doc = Document(id=str(uuid4()), library_id=lib_id, title=title, metadata=metadata or {})
        self.storage.save_document(doc)
        return doc

    def get_document(self, doc_id: str) -> Optional[Document]:
        return self.storage.load_document(doc_id)

    def list_documents(self, lib_id: str) -> List[Document]:
        return self.storage.load_documents_for_library(lib_id) or []

    def update_document(self, doc_id: str, updates: dict) -> Optional[Document]:
        if "library_id" in updates:
            raise ValueError("Changing document.library_id is not supported")
        return self.storage.update_document(doc_id, updates)

    def delete_document(self, lib_id: str, doc_id: str) -> bool:
        doc = self.get_document(doc_id)
        if not doc or doc.library_id != lib_id: 
            return False
        
        # Remove chunks from memory index
        idx = self.indexes.get(lib_id)
        if idx:
            chunks = self.storage.load_chunks_for_document(doc_id)
            for chunk in chunks:
                idx.remove_chunk(chunk.id)
        
        # Delete from disk
        self.storage.delete_chunks_for_document(doc_id)
        return self.storage.delete_document(doc_id)

    # -------- chunks --------
    def create_chunk(self, lib_id: str, doc_id: str, text: str, embedding: List[float], metadata: dict) -> Chunk:
        lib = self.get_library(lib_id)
        doc = self.get_document(doc_id)
        if not lib: raise KeyError("library")
        if not doc or doc.library_id != lib_id: raise KeyError("document")
        if len(embedding) != lib.dims: raise ValueError("dim mismatch")
        ch = Chunk(id=str(uuid4()), library_id=lib_id, document_id=doc_id, text=text, embedding=embedding, metadata=metadata or {})
        self.storage.save_chunk(ch)
        self._ensure_index(lib_id, lib.dims).add_chunk(ch)
        return ch
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.storage.load_chunk(chunk_id)

    def list_chunks(self, lib_id: str) -> List[Chunk]:
        lib = self.get_library(lib_id)
        if not lib: raise KeyError("library")
        return self.storage.load_chunks_for_library(lib_id) or []

    def update_chunk(self, lib_id: str, chunk_id: str, **updates) -> Optional[Chunk]:
        ch = self.storage.load_chunk(chunk_id)
        if not ch or ch.library_id != lib_id: return None
        if "embedding" in updates:
            lib = self.get_library(lib_id)
            if not lib or len(updates["embedding"]) != lib.dims: raise ValueError("dim mismatch")
        updated = self.storage.update_chunk(chunk_id, updates)
        if updated and "embedding" in updates:
            self._ensure_index(lib_id, self.get_library(lib_id).dims).update_chunk(chunk_id, updated)
        return updated

    def delete_chunk(self, lib_id: str, chunk_id: str) -> bool:
        ok = self.storage.delete_chunk(chunk_id)
        if ok:
            idx = self.indexes.get(lib_id)
            if idx: idx.remove_chunk(chunk_id)
        return ok

    # -------- other operations --------
    def search(self, lib_id: str, query: List[float], k: int, include_chunk: bool = False) -> List[SearchResult]:
        lib = self.get_library(lib_id)
        if not lib: raise KeyError("library")
        if len(query) != lib.dims: raise ValueError("dim mismatch")
        idx = self._ensure_index(lib_id, lib.dims)
        results = idx.search(query_embedding=query, k=k)
        if include_chunk:
            for r in results:
                r.chunk = self.storage.load_chunk(r.chunk_id)
        return results

    # TODO: might be useful for fast index loading from disk [Work in progress]
    # def rebuild_index(self, lib_id: str) -> None:
    #     lib = self.get_library(lib_id)
    #     if not lib: raise KeyError("library")
    #     new_idx = FlatIndex(dimension=lib.dims)
    #     for c in self.storage.load_chunks_for_library(lib_id) or []:
    #         new_idx.add_chunk(c)
    #     self.indexes[lib_id] = new_idx

    def _ensure_index(self, lib_id: str, dims: int) -> FlatIndex:
        """Ensure an index exists for a library: loads/create index for requested library if not exists"""
        idx = self.indexes.get(lib_id)
        if idx is None:
            idx = FlatIndex(dimension=dims)
            for c in self.storage.load_chunks_for_library(lib_id) or []:
                idx.add_chunk(c)
            self.indexes[lib_id] = idx
        return idx
