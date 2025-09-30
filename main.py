from fastapi import FastAPI
from .routes import vectors, health

app = FastAPI(
    title="Vector Store Service",
    version="1.1",
    description="A microservice to store, retrieve, and search vectors in ChromaDB"
)

# Register routers
app.include_router(health.router)
app.include_router(vectors.router)

@app.get("/")
def root():
    return {"message": "Vector Storage Service is running 🚀"}
