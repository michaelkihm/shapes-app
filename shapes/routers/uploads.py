import shutil
from typing import TypedDict

from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from ..dependencies import get_yolo

templates = Jinja2Templates(directory="shapes/templates")
router = APIRouter()


class PredResult(TypedDict):
    conf: float
    box: list[int]
    label: str


def extract_results(
    boxes: Boxes, names: dict[int, str], conf_thr=0.9
) -> list[PredResult]:
    predictions: list[PredResult] = []
    for box in boxes:
        conf = float(box.conf)
        if conf < conf_thr:
            continue
        predictions.append(
            {
                "conf": conf,
                "box": box.xywh[0].numpy().tolist(),
                "label": names[int(box.cls)],
            }
        )
    return predictions


@router.post("/upload", response_class=HTMLResponse)
async def get(
    request: Request, file: UploadFile = File(...), yolo: YOLO = Depends(get_yolo)
):
    # Save the uploaded file
    img_p = "imgs/uploaded_image.jpg"
    with open(img_p, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results: list[Results] = yolo(img_p)

    assert results[0].boxes
    predictions = extract_results(results[0].boxes, results[0].names)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "predictions": predictions,
        },
    )
