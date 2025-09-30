import uuid
import logging
import chromadb
from typing import List, Dict, Optional
from ..config import PERSIST_DIR, COLLECTION_NAME
from .utils import normalize_metadata

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.expected_dimension: Optional[int] = None

    def store_vector(self, vector: List[float], metadata: Dict):
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary.")

        metadata = normalize_metadata(metadata)

        if self.expected_dimension is None:
            self.expected_dimension = len(vector)
        elif len(vector) != self.expected_dimension:
            raise ValueError(f"Vector length mismatch. Expected {self.expected_dimension}, got {len(vector)}")

        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            embeddings=[vector],
            metadatas=[metadata],
            documents=[metadata.get("text", "")]
        )
        logging.info(f"✅ Vector stored with ID {doc_id}")
        return doc_id

    def get_all(self, limit: Optional[int] = None):
        return self.collection.get(limit=limit)

    def search(self, query_vector: List[float], top_k: int = 5):
        if self.expected_dimension and len(query_vector) != self.expected_dimension:
            raise ValueError(f"Query vector length mismatch. Expected {self.expected_dimension}")
        return self.collection.query(query_embeddings=[query_vector], n_results=top_k)

# Global instance
vector_store = VectorStore()
