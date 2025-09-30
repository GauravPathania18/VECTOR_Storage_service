import logging, json, asyncio, threading
import httpx
from .utils import normalize_metadata
from .vector_store import vector_store
from ..config import OLLAMA_URL

async def generate_metadata(text: str, model: str = "mistral") -> dict:
    async with httpx.AsyncClient(base_url=OLLAMA_URL) as client:
        prompt = f"""
        Analyze the following text and generate metadata in strict JSON format.
        Fields:
        - title
        - summary
        - keywords
        - category

        Text: {text[:1000]}
        """
        try:
            resp = await client.post("/api/generate", json={"model": model, "prompt": prompt, "stream": False}, timeout=60)
            resp.raise_for_status()
            raw_output = resp.json().get("response", "").strip()
            if not raw_output:
                return {"raw_output": ""}

            try:
                return json.loads(raw_output)
            except json.JSONDecodeError:
                start, end = raw_output.find("{"), raw_output.rfind("}")
                if start != -1 and end != -1:
                    return json.loads(raw_output[start:end+1])
                return {"raw_output": raw_output}
        except Exception as e:
            logging.error(f"[Metadata] Ollama request failed: {e}", exc_info=True)
            return {"error": str(e)}

async def update_metadata_in_chroma(doc_id: str, text: str, model: str = "mistral"):
    try:
        logging.info(f"🚀 Generating metadata for {doc_id}")
        metadata = await generate_metadata(text, model)
        metadata["text"] = text
        metadata["status"] = "ready"

        if isinstance(metadata.get("keywords"), list):
            metadata["keywords"] = ", ".join(metadata["keywords"])

        metadata = normalize_metadata(metadata)
        existing = vector_store.collection.get(ids=[doc_id])
        if not existing["ids"]:
            logging.warning(f"❌ Document {doc_id} not found")
            return

        old_meta = existing["metadatas"][0] if existing["metadatas"] else {}
        merged_meta = {**old_meta, **metadata}

        vector_store.collection.update(ids=[doc_id], metadatas=[merged_meta], documents=[text])
        logging.info(f"✅ Metadata updated for {doc_id}")
    except Exception as e:
        logging.error(f"[Metadata Update] Failed for {doc_id}: {e}", exc_info=True)

def run_update_metadata(doc_id: str, text: str, model: str = "mistral"):
    def runner():
        asyncio.run(update_metadata_in_chroma(doc_id, text, model))
    threading.Thread(target=runner, daemon=True).start()
