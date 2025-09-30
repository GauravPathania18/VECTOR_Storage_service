import logging
from fastapi import APIRouter, BackgroundTasks, HTTPException
from ..models import TextRequest, SearchRequest, QueryRequest
from ..services.vector_store import vector_store
from ..services.embedder import get_embedding
from ..services.metadata import run_update_metadata
from ..services.utils import clean_user_text, normalize_metadata

router = APIRouter()

@router.post("/add_text")
async def add_text(req: TextRequest, background_tasks: BackgroundTasks):
    try:
        clean_text = clean_user_text(req.text)
        if not clean_text:
            raise HTTPException(status_code=400, detail="Input text empty after cleaning.")

        vector = get_embedding(clean_text)
        metadata = req.metadata or {}
        metadata["text"] = req.text
        metadata["status"] = "pending"
        metadata = normalize_metadata(metadata)

        doc_id = vector_store.store_vector(vector, metadata)
        background_tasks.add_task(run_update_metadata, doc_id, clean_text, req.model)

        return {"status": "success", "id": doc_id, "text": req.text, "metadata_status": "pending"}
    except Exception as e:
        logging.error(f"[VectorStore] Failed: {e}")
        return {"status": "error", "message": str(e)}

@router.get("/vectors")
async def list_vectors():
    items = vector_store.collection.get()
    return {"status": "success", "results": [
        {"id": items["ids"][i], "document": items["documents"][i], "metadata": items["metadatas"][i]}
        for i in range(len(items["ids"]))
    ]}

@router.post("/search")
def search_vectors(req: SearchRequest):
    try:
        results = vector_store.search(req.vector, req.top_k or 5)
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/query_text")
def query_text(req: QueryRequest):
    try:
        query_vec = get_embedding(req.query)
        results = vector_store.search(query_vec, req.top_k or 5)
        return {"status": "success", "query": req.query, "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
