"""Prediction API routes."""
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from api_services.prediction_service import csv_to_json_preview, predict_csv, predict_single

router = APIRouter(prefix="/api/predict", tags=["prediction"])


@router.post("/single")
async def single_predict(data: dict):
    """Single enzyme-substrate prediction."""
    protein = data.get("protein", "").strip()
    smiles = data.get("smiles", "").strip()
    if not protein or not smiles:
        raise HTTPException(400, "protein and smiles are required")
    try:
        result = predict_single(protein, smiles)
        return {"status": "ok", **result}
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post("/batch")
async def batch_predict(
    file: UploadFile = File(...),
    seq_col: str = Form("Enzyme"),
    smiles_col: str = Form("Substrates"),
):
    """Batch CSV prediction. Returns JSON preview of results."""
    # Save upload to temp file
    tmp_in = Path(tempfile.mktemp(suffix=".csv"))
    try:
        with open(tmp_in, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    try:
        out_path = predict_csv(str(tmp_in), seq_col, smiles_col)
        preview = csv_to_json_preview(out_path)
        tmp_in.unlink(missing_ok=True)
        return {"status": "ok", **preview}
    except ValueError as e:
        tmp_in.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    except Exception as e:
        tmp_in.unlink(missing_ok=True)
        raise HTTPException(500, str(e))
