from pydantic import BaseModel
from typing import List, Dict, Optional

class TextRequest(BaseModel):
    text: str
    metadata: Optional[Dict] = None
    model: Optional[str] = "mistral"

class SearchRequest(BaseModel):
    vector: List[float]
    top_k: Optional[int] = 5

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
