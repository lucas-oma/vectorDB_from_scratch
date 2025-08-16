import os
from typing import Dict, List, Optional, Any
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ASCENDING

from .models import Chunk, Document, Library


class MongoStorage:
    """Handles MongoDB persistence for the vector db"""
    
    def __init__(self, connection_string: str = None, database_name: str = "vector_db"):
        if connection_string is None:
            connection_string = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        
        self.client = AsyncIOMotorClient(connection_string)
        self.db = self.client[database_name]
        
        # Collections
        self.libraries = self.db.libraries
        self.documents = self.db.documents
        self.chunks = self.db.chunks
        
    
    async def _create_indexes(self):
        await self.libraries.create_index([("id", ASCENDING)], unique=True)
        
        await self.documents.create_index([("id", ASCENDING)], unique=True)
        await self.documents.create_index([("library_id", ASCENDING)])
        
        await self.chunks.create_index([("id", ASCENDING)], unique=True)
        await self.chunks.create_index([("library_id", ASCENDING)])
        await self.chunks.create_index([("document_id", ASCENDING)])
    
    # -------- libraries --------
    async def save_library(self, library: Library) -> None:
        await self.libraries.replace_one(
            {"id": library.id}, 
            library.dict(), 
            upsert=True
        )
    
    async def load_library(self, library_id: str) -> Optional[Library]:
        doc = await self.libraries.find_one({"id": library_id})
        if doc:
            return Library(**doc)
        return None

    async def update_library(self, library_id: str, updates: Dict[str, Any]) -> Optional[Library]:
        # Remove None values and dimention change is not allowed
        clean_updates = {k: v for k, v in updates.items() 
                        if v is not None and k not in {"dims"}}
        
        if not clean_updates:
            return await self.load_library(library_id)
        
        result = await self.libraries.update_one(
            {"id": library_id},
            {"$set": clean_updates}
        )
        
        if result.modified_count > 0:
            return await self.load_library(library_id)
        return None
    
    async def load_all_libraries(self) -> Dict[str, Library]:
        cursor = self.libraries.find({})
        libraries = {}
        async for doc in cursor:
            libraries[doc["id"]] = Library(**doc)
        return libraries
    
    async def delete_library(self, library_id: str) -> bool:
        result = await self.libraries.delete_one({"id": library_id})
        return result.deleted_count > 0
    
    # -------- documents --------
    async def save_document(self, document: Document) -> None:
        await self.documents.replace_one(
            {"id": document.id}, 
            document.dict(), 
            upsert=True
        )
    
    async def load_documents_for_library(self, library_id: str) -> List[Document]:
        cursor = self.documents.find({"library_id": library_id})
        documents = []
        async for doc in cursor:
            documents.append(Document(**doc))
        return documents
    
    async def load_document(self, document_id: str) -> Optional[Document]:
        doc = await self.documents.find_one({"id": document_id})
        if doc:
            return Document(**doc)
        return None
    
    async def update_document(self, document_id: str, updates: Dict[str, Any]) -> Optional[Document]:
        # Remove None values and library_id change is not allowed
        clean_updates = {k: v for k, v in updates.items() 
                        if v is not None and k not in {"library_id"}}
        
        if not clean_updates:
            return await self.load_document(document_id)
        
        result = await self.documents.update_one(
            {"id": document_id},
            {"$set": clean_updates}
        )
        
        if result.modified_count > 0:
            return await self.load_document(document_id)
        return None
    
    async def delete_document(self, document_id: str) -> bool:
        result = await self.documents.delete_one({"id": document_id})
        return result.deleted_count > 0
    
    # -------- chunks --------
    async def save_chunk(self, chunk: Chunk) -> None:
        await self.chunks.replace_one(
            {"id": chunk.id}, 
            chunk.dict(), 
            upsert=True
        )
    
    async def load_chunks_for_library(self, library_id: str) -> List[Chunk]:
        cursor = self.chunks.find({"library_id": library_id})
        chunks = []
        async for doc in cursor:
            chunks.append(Chunk(**doc))
        return chunks
    
    async def load_chunk(self, chunk_id: str) -> Optional[Chunk]:
        doc = await self.chunks.find_one({"id": chunk_id})
        if doc:
            return Chunk(**doc)
        return None
    
    async def update_chunk(self, chunk_id: str, updates: Dict[str, Any]) -> Optional[Chunk]:
        # Remove None values
        clean_updates = {k: v for k, v in updates.items() if v is not None}
        
        if not clean_updates:
            return await self.load_chunk(chunk_id)
        
        result = await self.chunks.update_one(
            {"id": chunk_id},
            {"$set": clean_updates}
        )
        
        if result.modified_count > 0:
            return await self.load_chunk(chunk_id)
        return None
    
    async def delete_chunk(self, chunk_id: str) -> bool:
        result = await self.chunks.delete_one({"id": chunk_id})
        return result.deleted_count > 0
    
    async def load_chunks_for_document(self, document_id: str) -> List[Chunk]:
        cursor = self.chunks.find({"document_id": document_id})
        chunks = []
        async for doc in cursor:
            chunks.append(Chunk(**doc))
        return chunks

    async def delete_chunks_for_document(self, document_id: str) -> int:
        result = await self.chunks.delete_many({"document_id": document_id})
        return result.deleted_count

    async def delete_chunks_for_library(self, library_id: str) -> int:
        result = await self.chunks.delete_many({"library_id": library_id})
        return result.deleted_count

    async def delete_documents_for_library(self, library_id: str) -> int:
        result = await self.documents.delete_many({"library_id": library_id})
        return result.deleted_count

    async def close(self):
        self.client.close() 