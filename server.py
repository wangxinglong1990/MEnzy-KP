"""
DLKin API Server - FastAPI wrapper around the DLKin pipeline.
Loads models once at startup, serves prediction/clustering/docking endpoints.
"""
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is on PYTHONPATH and is the CWD
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(str(PROJECT_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all models and encoders at startup, held in memory."""
    print("Loading ESMC + SMILES Transformer + enzyme-model (ExtraTreesRegressor) ...")
    try:
        from api_services.model_loader import ModelService
        ModelService.get_instance()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"WARNING: Model loading failed - {e}")
        print("Prediction endpoints will return 503 until models are available.")
    yield


app = FastAPI(title="DLKin API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    from api_services.model_loader import ModelService
    try:
        ModelService.get_instance()
        return {"status": "ok", "models_loaded": True}
    except Exception:
        return {"status": "initializing", "models_loaded": False}


# Import API route modules (registers routers)
from api_routes import predict  # noqa: E402
app.include_router(predict.router)

# Serve frontend in production
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")


if __name__ == "__main__":
    import os
    os.chdir(str(PROJECT_ROOT))  # ESMC library resolves relative data paths from CWD
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=9090, reload=True)
