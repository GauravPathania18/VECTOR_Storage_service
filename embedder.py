import requests
import logging
from fastapi import HTTPException
from ..config import EMBEDDER_URL

def get_embedding(text: str):
    try:
        payload = {"texts": [text]}
        resp = requests.post(EMBEDDER_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "items" not in data or not data["items"]:
            raise ValueError("No embeddings returned")
        return data["items"][0]["vector"]
    except requests.exceptions.RequestException as e:
        logging.error(f"[Embedder] HTTP error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedder error: {e}")
