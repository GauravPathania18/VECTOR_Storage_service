# store_service.py

import chromadb
import uuid
import logging
import requests
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
import httpx
import threading
import asyncio

# -------------------------------------------------
# FastAPI App Initialization
# -------------------------------------------------
app = FastAPI(
    title="Vector Store Service",
    version="1.1",
    description="A microservice to store, retrieve, and search vectors in ChromaDB"
)

'''# Initialize Chroma client
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("documents")
'''

# -------------------------------------------------
# Logging Setup
# -------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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

        # Normalize metadata always before storing
        metadata = normalize_metadata(metadata)
        
        # Check vector dimension
        if self.expected_dimension is None:
            self.expected_dimension = len(vector)
        elif len(vector) != self.expected_dimension:
            raise ValueError(f"Vector must have length {self.expected_dimension}, got {len(vector)}.")

        doc_id = str(uuid.uuid4())
        self.collection.add(
            ids=[doc_id],
            embeddings=[vector],
            metadatas=[metadata],
            documents=[metadata.get("text", "")]  # store text if available
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
    

    # # 🔹 safe reset
    # def reset_collection_soft(self):
    #     """Delete all vectors from collection without dropping it."""
    #     page = self.collection.get(include=[], limit=None)
    #     ids = page.get("ids", [])
    #     if ids:
    #         self.collection.delete(ids=ids)
    #     self.expected_dimension = None
    #     logging.info("🧹 Soft reset: deleted %d items", len(ids))
    #     return len(ids)
    

# Create a single global instance of VectorStore
vector_store = VectorStore()


# --- Reset collection for fresh embeddings (temporary, for testing) ---
# all_docs = vector_store.collection.get()
# ids_to_delete = all_docs['ids']  # list of all stored IDs
# if ids_to_delete:
#     vector_store.collection.delete(ids=ids_to_delete)
# vector_store.expected_dimension = None

# -------------------------------------------------
# Request Models
# -------------------------------------------------
'''class StoreRequest(BaseModel):
    vector: List[float]   # e.g. [0.12, 0.98, -0.33, ...]
    metadata: Optional[Dict] = None  # e.g. {"doc": "my note"}'''

class TextRequest(BaseModel):
    '''so the vector store can:
           Receive raw text + metadata.
           Call the embedder internally to generate the vector.
           Store the vector in ChromaDB.'''
    text: str
    metadata: Optional[Dict] = None 
    model: Optional[str] = "mistral"  # metadata generator model
    
class SearchRequest(BaseModel):
    vector: List[float]
    top_k: Optional[int] = 5

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

# -------------------------------------------------
# Text Cleaning
# -------------------------------------------------
def clean_user_text(raw_text: str) -> str:
    """
    Clean raw user input:
    - Replace newlines with spaces
    - Strip extra whitespace
    """
    if not raw_text:
        return ""
    return " ".join(raw_text.split())

# -------------------------------------------------
# External Embedder API
# -------------------------------------------------
EMBEDDER_URL = "http://127.0.0.1:8001/embed"

def get_embedding(text: str) -> List[float]:
    """
    Sends text to the embedder service and returns the first embedding vector.
    """
    try:
        # The embedder expects a list of texts
        payload = {"texts": [text]}
        resp = requests.post(EMBEDDER_URL, json=payload)
        resp.raise_for_status()

        data = resp.json()

        # Validate response structure
        if "items" not in data or not data["items"]:
            raise ValueError("No embeddings returned by the embedder.")

        # Return the first vector
        return data["items"][0]["vector"]

    except requests.exceptions.RequestException as e:
        logging.error(f"[Embedder] HTTP request failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedder request error: {e}")
    except (KeyError, ValueError) as e:
        logging.error(f"[Embedder] Invalid response format: {e}, response={resp.text}")
        raise HTTPException(status_code=500, detail=f"Embedder response error: {e}")


def normalize_metadata(meta: dict) -> dict:
    normalized = {}
    for k, v in meta.items():
        if isinstance(v, list):
            # store lists as comma-separated string
            normalized[k] = ", ".join(map(str, v))
        elif isinstance(v, dict):
            # if nested dict, flatten it too
            normalized[k] = json.dumps(v)
        else:
            normalized[k] = v
        
    return normalized
    
# -------------------------------------------------
# Metadata Generator (via Ollama + Mistral/LLaMA)
# -------------------------------------------------

# create one async client to reuse connections
# client = httpx.AsyncClient(base_url="http://localhost:11434")

async def generate_metadata(text: str, model: str = "mistral") -> dict:
    async with httpx.AsyncClient(base_url="http://localhost:11434") as client:

        text = text[:1000] #truncate long inputs

        prompt = f"""
        Analyze the following text and generate metadata in strict JSON format.
        Fields:
        - title: short descriptive title
        - summary: concise summary
        - keywords: list of 3-5 important keywords
        - category: broad category/topic

        Text:
        {text}
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }

        try:
            resp = await client.post("/api/generate", json=payload , timeout=60)
            resp.raise_for_status()
            j = resp.json() #safely retrieve response text
            raw_output = j.get("response", "") if isinstance(j, dict) else ""
            raw_output = (raw_output or "").strip()
            if not raw_output:
                logging.warning("[Metadata] Ollama returned empty response.")
                return {"raw_output": ""}
            # Try to parse JSON from model output
            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                # If model includes extra text, try to extract a JSON substring
                start = raw_output.find("{")
                end = raw_output.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        return json.loads(raw_output[start:end+1])
                    except Exception as e:
                        logging.warning(f"[Metadata] failed to parse extracted JSON: {e}")
                        return {"raw_output": raw_output}
                # fallback
                return {"raw_output": raw_output}
        except Exception as e:
            logging.error(f"[Metadata] Ollama request failed: {e}", exc_info=True)
            return {"error": str(e), "raw_output": ""}


    

# ---- Background Task ----
async def update_metadata_in_chroma(doc_id: str, text: str, model: str = "mistral"):
    
    '''Background task that generates metadata and updates ChromaDB.
    This function is async and will be scheduled with asyncio.create_task via BackgroundTasks.'''
    try:
        logging.info(f"🚀 Starting metadata generation for {doc_id}")
        
        metadata = await generate_metadata(text, model)
        metadata["text"] = text
        metadata["status"] = "ready"
        # Ensure keywords are string
        if isinstance(metadata.get("keywords"), list):
            metadata["keywords"] = ", ".join(metadata["keywords"])
        # Normalize before storing
        metadata = normalize_metadata(metadata)

        # 🔹 Fetch existing doc to preserve fields (like "text", "status")
        existing = vector_store.collection.get(ids=[doc_id])
        if not existing["ids"]:
            logging.warning(f"❌ Document {doc_id} not found for metadata update")
            return
        
        old_meta = existing["metadatas"][0] if existing["metadatas"] else {}

        # Merge old + new
        merged_meta = {**old_meta, **metadata,}

        vector_store.collection.update(
            ids=[doc_id],
            metadatas=[merged_meta],
            documents=[text]
        )
        logging.info(f"✅ Metadata updated for {doc_id}")
    except Exception as e:
        logging.error(f"[Metadata Update] Failed for {doc_id}: {e}", exc_info=True)



def run_update_metadata(doc_id: str, text: str, model: str = "mistral"):
    """Run async metadata update in its own thread."""
    asyncio.run(update_metadata_in_chroma(doc_id, text, model))
    threading.Thread(target=runner, daemon=True).start()


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------

@app.get("/")
def root():
    return {"message": "Vector Storage Service is running 🚀"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/add_text")
async def add_text(req: TextRequest,background_tasks: BackgroundTasks):
    """
    Accept raw text + optional metadata, generate embedding via embedder, and store in ChromaDB.
    """
    try:
        # 🧹 Clean raw text before embedding
        clean_text = clean_user_text(req.text)
        
        if not clean_text:
            raise HTTPException(status_code=400, detail="Input text is empty after cleaning.")  
        logging.info(f"📝 /add_text called with text='{req.text[:30]}...'")     
        
        # 🔹 Generate embedding via external embedder service
        # synchronous embedder call
        vector = get_embedding(clean_text)# <-- calls embedder
            
        

        # Merge user-provided metadata with generated metadata
        metadata = req.metadata or {}
        metadata["text"] = req.text  # always store original text
        metadata["status"] = "pending"

        # ✅ Normalize before storing
        metadata = normalize_metadata(metadata)
        '''# Store vector with metadata
            metadata = req.metadata or {"text": req.text}'''
        
        # 🔹 Store vector immediately
        doc_id = vector_store.store_vector(vector, metadata)

        
        # 🔹Queue metadata generation in background(runs after response is sent)
        background_tasks.add_task(run_update_metadata, doc_id, clean_text, req.model)

        return {
                "status": "success", 
                "id": doc_id, 
                "text": req.text,
                "metadata_status":"pending"
            }
    except Exception as e:
        logging.error(f"[VectorStore] Failed to store vector: {e}")
        return {"status": "error", "message": str(e)}
    
    
@app.get("/vectors")
async def list_vectors():
    items = vector_store.collection.get()
    results = []

    for i in range(len(items["ids"])):
        results.append({
            "id": items["ids"][i],
            "document": items["documents"][i],
            "metadata": items["metadatas"][i],   # ✅ include metadata
            # "embedding": None   # keep excluded (too large)
        })
    return {"status": "success", "results": results}


    


# ✅ Core search endpoint (low-level: takes vector directly)
@app.post("/search")
def search_vectors(req: SearchRequest):
    try:
        logging.info(f"🔍 /search called with top_k={req.top_k}")
        results = vector_store.search(req.vector, req.top_k or 5)
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 🔹 Query text (embed + similarity search)
@app.post("/query_text")
def query_text(req: QueryRequest):
    """
    High-level search: accepts query text, generates embedding internally, then searches.
    """
    try:
        logging.info(f"🔍 /query_text called with top_k={req.top_k}, query='{req.query[:30]}...'")
        query_vec = get_embedding(req.query)
        results = vector_store.search(query_vec, req.top_k or 5)
        return {
            "status": "success",
            "query": req.query,
            "results": results
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


'''@app.post("/reset_soft")
def reset_collection_soft():
    try:
        deleted = vector_store.reset_soft()
        return {"status": "success", "message": f"Soft reset: {deleted} items deleted."}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

@app.post("/reset_hard")
def reset_collection_hard():
    """
    Completely reset ChromaDB collection (including wrong dimension).
    """
    try:
        # Delete by name used in __init__
        vector_store.client.delete_collection(name="personal_llm")
        # Recreate same collection
        vector_store.collection = vector_store.client.create_collection(name="personal_llm")

        vector_store.expected_dimension = None
        return {"status": "success", "message": "Hard reset: collection deleted and recreated."}
    except Exception as e:
        return {"status": "error", "message": str(e)}'''


