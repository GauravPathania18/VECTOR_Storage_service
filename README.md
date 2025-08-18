# VECTOR_Storage_service
Vector Store Service –Vector database microservice for storing, retrieving, and searching embeddings.
Built with FastAPI and ChromaDB for use in AI apps, semantic search, and LLM pipelines.


#🚀 Features

Store vectors with associated metadata

Retrieve all stored vectors (with optional limit)

Semantic search using vector similarity (top_k results)

Persistent storage (vectors survive restarts)

Simple REST API with FastAPI docs

#⚙️ Setup

1. Clone the repo:
bash ```
git clone https://github.com/<your-username>/vector-store-service.git
cd vector-store-service```


2. Install dependencies:
bash```
pip install -r requirements.txt```


3. Run the service:
bash```
uvicorn store_service:app --reload --host 0.0.0.0 --port 8001```


4. Check health/docs:

-->Swagger UI: http://localhost:8001/docs

-->ReDoc: http://localhost:8001/redoc

📡 API Endpoints
POST /store – Store a vector

Request:

{
  "vector": [0.12, 0.98, -0.33],
  "metadata": { "doc": "my note" }
}


Response:

{
  "status": "success",
  "id": "uuid",
  "message": "Vector stored successfully."
}

GET /vectors – Fetch stored vectors

Optional Query Param: limit

Response:

{
  "status": "success",
  "data": {
    "ids": ["uuid1", "uuid2"],
    "embeddings": [[...], [...]],
    "metadatas": [{"doc": "my note"}, {"doc": "another"}]
  }
}

POST /search – Search vectors by similarity

Request:

{
  "vector": [0.1, 0.9, -0.3],
  "top_k": 3
}


Response:

{
  "status": "success",
  "results": {
    "ids": [["uuid1", "uuid2"]],
    "distances": [[0.12, 0.34]],
    "metadatas": [[{"doc": "closest match"}, {"doc": "second match"}]]
  }
}

⚙️ Environment Variables

You can configure storage:

export PERSIST_DIR=./chroma_store
export COLLECTION_NAME=personal_llm

📖 About

A lightweight vector storage API for AI pipelines.
Pairs well with the Personal LLM Embedding Service.

📜 License

MIT License
