import requests
import json
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables from .env file (in current directory)
load_dotenv("../.env")

# Cohere API configuration from environment variables
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "your_key_here")
COHERE_EMBED_URL = os.getenv("COHERE_EMBED_URL", "https://api.cohere.ai/v1/embed")


def generate_embedding(text: str) -> List[float]:
    """Generate embedding for text using Cohere API."""
        
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "texts": [text],
        "model": "embed-english-v3.0",
        "input_type": "search_document"
    }
    
    response = requests.post(COHERE_EMBED_URL, headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    return result["embeddings"][0]
    
    


def generate_test_chunks(library_id: str, document_id: str, num_chunks: int = 5) -> List[Dict]:
    """Generate test chunks with real embeddings."""
    
    # Sample texts for testing - MongoDB-focused content
    sample_texts = [
        "MongoDB is a NoSQL database that stores data in flexible, JSON-like documents.",
        "Vector databases enable efficient similarity search for high-dimensional data like embeddings.",
        "FastAPI provides modern, fast web framework for building APIs with Python.",
        "Docker containers package applications with their dependencies for consistent deployment.",
        "Asynchronous programming with asyncio allows non-blocking I/O operations in Python.",
        "Pydantic provides data validation using Python type annotations.",
        "Motor is an async MongoDB driver for Python that works with asyncio.",
        "Vector similarity search finds the most similar items in high-dimensional spaces.",
        "Microservices architecture decomposes applications into small, independent services.",
        "Kubernetes orchestrates containerized applications across multiple hosts."
    ]
    
    chunks = []
    
    for i in range(num_chunks):
        # Select a random text or use all if num_chunks > len(sample_texts)
        text = sample_texts[i % len(sample_texts)]
        
        # Generate embedding
        embedding = generate_embedding(text)
        
        chunk = {
            "id": f"chunk_{document_id}_{i+1}",
            "document_id": document_id,
            "library_id": library_id,
            "text": text,
            "embedding": embedding,
            "metadata": {
                "chunk_index": str(i + 1),
                "document_id": document_id,
                "library_id": library_id,
                "generated": "true",
                "topic": sample_texts[i % len(sample_texts)].split()[0].lower()
            }
        }
        
        chunks.append(chunk)
    
    return chunks


def generate_test_data():
    """Generate complete test data for the MongoDB vector database."""
    
    # Test libraries with different index types
    libraries = [
        {
            "id": "tech_library",
            "name": "Technology Library",
            "dims": 1024,  # Cohere embed-english-v3.0 dimension
            "index_type": "flat",
            "metadata": {
                "category": "technology", 
                "description": "Tech-related documents",
                "created_by": "test_suite"
            }
        },
        {
            "id": "science_library", 
            "name": "Science Library",
            "dims": 1024,  # Cohere embed-english-v3.0 dimension
            "index_type": "flat",
            "metadata": {
                "category": "science", 
                "description": "Scientific documents",
                "created_by": "test_suite"
            }
        }
    ]
    
    # Test documents
    documents = [
        {
            "id": "ml_doc",
            "library_id": "tech_library",
            "title": "Machine Learning Guide",
            "metadata": {
                "topic": "machine_learning", 
                "author": "AI Expert",
                "created_by": "test_suite"
            }
        },
        {
            "id": "nlp_doc",
            "library_id": "tech_library", 
            "title": "NLP Fundamentals",
            "metadata": {
                "topic": "nlp", 
                "author": "Language Expert",
                "created_by": "test_suite"
            }
        },
        {
            "id": "physics_doc",
            "library_id": "science_library",
            "title": "Physics Principles",
            "metadata": {
                "topic": "physics", 
                "author": "Physicist",
                "created_by": "test_suite"
            }
        }
    ]
    
    # Generate chunks for each document
    all_chunks = []
    
    for doc in documents:
        chunks = generate_test_chunks(
            library_id=doc["library_id"],
            document_id=doc["id"],
            num_chunks=3
        )
        all_chunks.extend(chunks)
    
    return {
        "libraries": libraries,
        "documents": documents,
        "chunks": all_chunks
    }


if __name__ == "__main__":

    print("Generating test data with Cohere API embeddings...")
    test_data = generate_test_data()
    
    # Save to JSON file for easy testing
    output_file = "tests-mongo/test_data.json"
    with open(output_file, "w") as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Generated test data:")
    print(f"- {len(test_data['libraries'])} libraries")
    print(f"- {len(test_data['documents'])} documents") 
    print(f"- {len(test_data['chunks'])} chunks")
    print(f"Saved to {output_file}")