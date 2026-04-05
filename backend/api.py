import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from .pipeline import pipeline
from .utils.image import load_image_from_bytes, image_to_base64, save_result
from .utils.config import cfg

_executor = ThreadPoolExecutor(max_workers=1)

app = FastAPI(title="Clothes Swap API", version="2.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    return pipeline.status()


@app.post("/swap/garment")
async def swap_garment(
    person_image:  UploadFile = File(...),
    garment_image: UploadFile = File(...),
    garment_type:  str = Form("upper"),
):
    person  = load_image_from_bytes(await person_image.read())
    garment = load_image_from_bytes(await garment_image.read())
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor, pipeline.run_garment_swap, person, garment, garment_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _build_response(result)


@app.post("/swap/text")
async def swap_text(
    person_image: UploadFile = File(...),
    prompt:       str = Form(...),
    garment_type: str = Form("upper"),
):
    person = load_image_from_bytes(await person_image.read())
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            _executor, pipeline.run_text_swap, person, prompt, garment_type
        )
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return _build_response(result)


def _build_response(result: Image.Image) -> JSONResponse:
    payload = {}
    if cfg.storage.return_base64:
        payload["image_base64"] = image_to_base64(result)
    if cfg.storage.save_results:
        path = save_result(result)
        payload["saved_path"] = str(path)
    return JSONResponse(payload)
