# store_service.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Optional
import chromadb
import uuid
import logging

# -------------------------------------------------
# FastAPI App Initialization
# -------------------------------------------------
app = FastAPI(
    title="Vector Store Service",
    version="1.1",
    description="A microservice to store, retrieve, and search vectors in ChromaDB"
)

# -------------------------------------------------
# Logging Setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
# Vector Store Logic
# -------------------------------------------------
class VectorStore:
    def __init__(self, persist_directory="./chroma_store", collection_name="personal_llm"):
        """
        Initialize ChromaDB with persistence.
        Data will be saved in the given folder so it survives restarts.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        # Optional: define expected embedding dimension (set after first vector stored)
        self.expected_dimension: Optional[int] = None

    def store_vector(self, vector: List[float], metadata: Dict):
        """
        Store a single vector along with metadata.
        Each vector is given a unique UUID as its ID.
        """
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary.")

        # Check vector dimension
        if self.expected_dimension is None:
            self.expected_dimension = len(vector)
        elif len(vector) != self.expected_dimension:
            raise ValueError(f"Vector must have length {self.expected_dimension}, got {len(vector)}.")

        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            embeddings=[vector],
            metadatas=[metadata]
        )
        logging.info(f"✅ Vector stored with ID {doc_id}")
        return doc_id

    def get_all(self, limit: Optional[int] = None):
        """
        Retrieve all vectors, optionally limited to 'limit' results.
        """
        return self.collection.get(limit=limit)

    def search(self, query_vector: List[float], top_k: int = 5):
        """
        Search for the top_k most similar vectors.
        """
        if self.expected_dimension and len(query_vector) != self.expected_dimension:
            raise ValueError(f"Query vector must have length {self.expected_dimension}, got {len(query_vector)}.")

        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

# Create a single global instance of VectorStore
vector_store = VectorStore()

# -------------------------------------------------
# Request Models
# -------------------------------------------------
class StoreRequest(BaseModel):
    vector: List[float]   # e.g. [0.12, 0.98, -0.33, ...]
    metadata: Optional[Dict] = None  # e.g. {"doc": "my note"}

class SearchRequest(BaseModel):
    vector: List[float]
    top_k: Optional[int] = 5

# -------------------------------------------------
# API Endpoints
# -------------------------------------------------
@app.post("/store")
def store_vector(req: StoreRequest):
    """
    Store a vector + metadata in ChromaDB.
    """
    try:
        doc_id = vector_store.store_vector(req.vector, req.metadata or {})
        return {"status": "success", "id": doc_id, "message": "Vector stored successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/vectors")
def get_vectors(limit: Optional[int] = None):
    """
    Fetch stored vectors (with optional limit).
    """
    try:
        results = vector_store.get_all(limit=limit)
        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/search")
def search_vectors(req: SearchRequest):
    """
    Search for the most similar vectors given a query.
    """
    try:
        results = vector_store.search(req.vector, req.top_k or 5)
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
