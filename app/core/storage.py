"""
Storage layer for persisting chunks, documents, and libraries to disk.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from .models import Chunk, Document, Library


class DiskStorage:
    """Handles disk persistence for the vector database."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.getenv("DATA_DIR", "db")
        self.data_dir = Path(data_dir)
        
        # Create all necessary directories
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / "snapshots").mkdir(exist_ok=True)
        
        # File paths
        self.libraries_file = self.data_dir / "libraries.json"
        self.documents_file = self.data_dir / "documents.jsonl"
        self.chunks_file = self.data_dir / "chunks.jsonl"
        
        self._init_files()
    
    def _init_files(self):
        """Initialize storage files if they don't exist."""
        if not self.libraries_file.exists():
            self.libraries_file.write_text("{}")
        
        if not self.documents_file.exists():
            self.documents_file.touch()
        
        if not self.chunks_file.exists():
            self.chunks_file.touch()
    
    # -------- libraries --------
    def save_library(self, library: Library) -> None:
        """Save a library to disk."""
        libraries = self._load_libraries()
        libraries[library.id] = library.dict()
        self._save_libraries(libraries)
    
    def load_library(self, library_id: str) -> Optional[Library]:
        """Load a library from disk."""
        libraries = self._load_libraries()
        if library_id in libraries:
            return Library(**libraries[library_id])
        return None

    def update_library(self, library_id: str, updates: Dict[str, Any]) -> Optional[Library]:
        libs = self._load_libraries()
        lib_dict = libs.get(library_id)
        if not lib_dict:
            return None

        # Changing dims is not allowed
        disallowed = {"dims"}
        for k, v in updates.items():
            if v is None or k in disallowed:
                continue
            lib_dict[k] = v

        libs[library_id] = lib_dict
        self._save_libraries(libs)
        return Library(**lib_dict)

    
    def load_all_libraries(self) -> Dict[str, Library]:
        """Load all libraries from disk."""
        libraries = self._load_libraries()
        return {lib_id: Library(**lib_data) for lib_id, lib_data in libraries.items()}
    
    def delete_library(self, library_id: str) -> bool:
        """Delete a library from disk."""
        libraries = self._load_libraries()
        if library_id in libraries:
            del libraries[library_id]
            self._save_libraries(libraries)
            return True
        return False
    
    def _load_libraries(self) -> Dict:
        """Load libraries JSON file."""
        return json.loads(self.libraries_file.read_text())
    
    def _save_libraries(self, libraries: Dict) -> None:
        """Save libraries to JSON file."""
        self.libraries_file.write_text(json.dumps(libraries, indent=2))
    
    # -------- documents --------
    def save_document(self, document: Document) -> None:
        """Save a document to disk (JSONL format)."""
        with open(self.documents_file, 'a') as f:
            f.write(json.dumps(document.dict()) + '\n')
    
    def load_documents_for_library(self, library_id: str) -> List[Document]:
        """Load all documents for a specific library."""
        documents = []
        with open(self.documents_file, 'r') as f:
            for line in f:
                if line.strip():
                    doc_data = json.loads(line)
                    if doc_data['library_id'] == library_id:
                        documents.append(Document(**doc_data))
        return documents
    
    def load_document(self, document_id: str) -> Optional[Document]:
        """Load a specific document by ID."""
        with open(self.documents_file, 'r') as f:
            for line in f:
                if line.strip():
                    doc_data = json.loads(line)
                    if doc_data['id'] == document_id:
                        return Document(**doc_data)
        return None
    
    def update_document(self, document_id: str, updates: Dict[str, Any]) -> Optional[Document]:
        """Update a document with the given updates."""
        temp_file = self.documents_file.with_suffix('.tmp')
        updated_doc = None
        
        with open(self.documents_file, 'r') as infile, open(temp_file, 'w') as outfile:
            for line in infile:
                if line.strip():
                    doc_data = json.loads(line)
                    if doc_data['id'] == document_id:
                        for key, value in updates.items():
                            if value is not None:
                                doc_data[key] = value
                        updated_doc = Document(**doc_data)
                        outfile.write(json.dumps(updated_doc.dict()) + '\n')
                    else:
                        outfile.write(line)
        
        if updated_doc:
            temp_file.replace(self.documents_file)
        
        return updated_doc
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document from disk (rewrite file without the document)."""
        temp_file = self.documents_file.with_suffix('.tmp')
        found = False
        
        with open(self.documents_file, 'r') as infile, open(temp_file, 'w') as outfile:
            for line in infile:
                if line.strip():
                    doc_data = json.loads(line)
                    if doc_data['id'] != document_id:
                        outfile.write(line)
                    else:
                        found = True
        
        if found:
            temp_file.replace(self.documents_file)
        
        return found
    
    # -------- chunks --------
    def save_chunk(self, chunk: Chunk) -> None:
        """Save a chunk to disk (JSONL format)."""
        with open(self.chunks_file, 'a') as f:
            f.write(json.dumps(chunk.dict()) + '\n')
    
    def load_chunks_for_library(self, library_id: str) -> List[Chunk]:
        """Load all chunks for a specific library."""
        chunks = []
        with open(self.chunks_file, 'r') as f:
            for line in f:
                if line.strip():
                    chunk_data = json.loads(line)
                    if chunk_data['library_id'] == library_id:
                        chunks.append(Chunk(**chunk_data))
        return chunks
    
    def load_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Load a specific chunk by ID."""
        with open(self.chunks_file, 'r') as f:
            for line in f:
                if line.strip():
                    chunk_data = json.loads(line)
                    if chunk_data['id'] == chunk_id:
                        return Chunk(**chunk_data)
        return None
    
    def update_chunk(self, chunk_id: str, updates: Dict[str, Any]) -> Optional[Chunk]:
        """Update a chunk with the given updates."""
        temp_file = self.chunks_file.with_suffix('.tmp')
        updated_chunk = None
        
        with open(self.chunks_file, 'r') as infile, open(temp_file, 'w') as outfile:
            for line in infile:
                if line.strip():
                    chunk_data = json.loads(line)
                    if chunk_data['id'] == chunk_id:
                        for key, value in updates.items():
                            if value is not None:
                                chunk_data[key] = value
                        updated_chunk = Chunk(**chunk_data)
                        outfile.write(json.dumps(updated_chunk.dict()) + '\n')
                    else:
                        outfile.write(line)
        
        if updated_chunk:
            temp_file.replace(self.chunks_file)
        
        return updated_chunk
    
    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a chunk from disk (rewrite file without the chunk)."""
        temp_file = self.chunks_file.with_suffix('.tmp')
        found = False
        
        with open(self.chunks_file, 'r') as infile, open(temp_file, 'w') as outfile:
            for line in infile:
                if line.strip():
                    chunk_data = json.loads(line)
                    if chunk_data['id'] != chunk_id:
                        outfile.write(line)
                    else:
                        found = True
        
        if found:
            temp_file.replace(self.chunks_file)
        
        return found
    
    def load_chunks_for_document(self, document_id: str) -> List[Chunk]:
        """Load all chunks for a specific document."""
        chunks = []
        with open(self.chunks_file, 'r') as f:
            for line in f:
                if line.strip():
                    chunk_data = json.loads(line)
                    if chunk_data['document_id'] == document_id:
                        chunks.append(Chunk(**chunk_data))
        return chunks

    def delete_chunks_for_document(self, document_id: str) -> int:
        """Delete all chunks for a specific document. Returns number of chunks deleted."""
        temp_file = self.chunks_file.with_suffix('.tmp')
        deleted_count = 0
        
        with open(self.chunks_file, 'r') as infile, open(temp_file, 'w') as outfile:
            for line in infile:
                if line.strip():
                    chunk_data = json.loads(line)
                    if chunk_data['document_id'] != document_id:
                        outfile.write(line)
                    else:
                        deleted_count += 1
        
        if deleted_count > 0:
            temp_file.replace(self.chunks_file)
        
        return deleted_count

    def delete_chunks_for_library(self, library_id: str) -> int:
        """Delete all chunks for a specific library. Returns number of chunks deleted."""
        temp_file = self.chunks_file.with_suffix('.tmp')
        deleted_count = 0
        
        with open(self.chunks_file, 'r') as infile, open(temp_file, 'w') as outfile:
            for line in infile:
                if line.strip():
                    chunk_data = json.loads(line)
                    if chunk_data['library_id'] != library_id:
                        outfile.write(line)
                    else:
                        deleted_count += 1
        
        if deleted_count > 0:
            temp_file.replace(self.chunks_file)
        
        return deleted_count

    def delete_documents_for_library(self, library_id: str) -> int:
        """Delete all documents for a specific library. Returns number of documents deleted."""
        temp_file = self.documents_file.with_suffix('.tmp')
        deleted_count = 0
        
        with open(self.documents_file, 'r') as infile, open(temp_file, 'w') as outfile:
            for line in infile:
                if line.strip():
                    doc_data = json.loads(line)
                    if doc_data['library_id'] != library_id:
                        outfile.write(line)
                    else:
                        deleted_count += 1
        
        if deleted_count > 0:
            temp_file.replace(self.documents_file)
        
        return deleted_count

    def delete_snapshot_files(self, library_id: str) -> bool:
        """Delete snapshot files for a library if they exist."""
        snapshot_dir = self.data_dir / "snapshots"
        vectors_file = snapshot_dir / f"vectors_{library_id}.npy"
        ids_file = snapshot_dir / f"ids_{library_id}.json"
        
        deleted = False
        if vectors_file.exists():
            vectors_file.unlink()
            deleted = True
        if ids_file.exists():
            ids_file.unlink()
            deleted = True
        
        return deleted
    
    # TODO:Snapshot operations for fast startup
    # def save_library_snapshot(self, library_id: str, vectors: np.ndarray, chunk_ids: List[str]) -> None:
    #     """Save a library's vectors and IDs for fast startup."""
    #     snapshot_dir = self.data_dir / "snapshots"
    #     snapshot_dir.mkdir(exist_ok=True)
        
    #     # Save vectors as numpy array
    #     vectors_file = snapshot_dir / f"vectors_{library_id}.npy"
    #     np.save(vectors_file, vectors)
        
    #     # Save chunk IDs
    #     ids_file = snapshot_dir / f"ids_{library_id}.json"
    #     with open(ids_file, 'w') as f:
    #         json.dump(chunk_ids, f)
    
    # def load_library_snapshot(self, library_id: str) -> Optional[tuple[np.ndarray, List[str]]]:
    #     """Load a library's vectors and IDs from snapshot."""
    #     snapshot_dir = self.data_dir / "snapshots"
    #     vectors_file = snapshot_dir / f"vectors_{library_id}.npy"
    #     ids_file = snapshot_dir / f"ids_{library_id}.json"
        
    #     if vectors_file.exists() and ids_file.exists():
    #         vectors = np.load(vectors_file)
    #         with open(ids_file, 'r') as f:
    #             chunk_ids = json.load(f)
    #         return vectors, chunk_ids
        
    #     return None 