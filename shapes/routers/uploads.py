import os
import shutil
from typing import TypedDict

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from ..dependencies import get_yolo

templates = Jinja2Templates(directory="shapes/templates")
router = APIRouter()


class PredResult(TypedDict):
    conf: float
    box: list[int]  # box x,y,w,h
    label: str


def convert_xywh_to_box(xywh_box: Tensor) -> list[int]:
    """
    Convert ultralytics xywh bounding box [x_center, y_center, width, height]
    to [x,y,w,h]
    """
    x_c, y_c, w, h = xywh_box[0].numpy().tolist()
    return [int(x_c - w / 2), int(y_c - h / 2), int(w), int(h)]


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
                "conf": round(conf, 2),
                "box": convert_xywh_to_box(box.xywh),
                "label": names[int(box.cls)],
            }
        )
    return predictions


def crop_detections(detections: list[PredResult], query_img: np.ndarray) -> list[str]:
    paths: list[str] = []
    for i, det in enumerate(detections):
        x, y, w, h = det["box"]
        path = f"shapes/static/imgs/detection_{i}.jpg"
        paths.append(path)
        cv2.imwrite(
            path,
            query_img[y : y + h, x : x + w],
        )

    return paths


@router.post("/upload", response_class=HTMLResponse)
async def model_inference(
    request: Request, file: UploadFile = File(...), yolo: YOLO = Depends(get_yolo)
):
    # Save the uploaded file
    img_p = "shapes/static/imgs/uploaded_image.jpg"
    with open(img_p, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results: list[Results] = yolo(img_p)

    assert results[0].boxes
    predictions = extract_results(results[0].boxes, results[0].names)

    print(
        [
            {**pred, "img": os.path.basename(det)}
            for pred, det in zip(
                predictions, crop_detections(predictions, cv2.imread(img_p))
            )
        ]
    )
    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "predictions": [
                {**pred, "img": os.path.basename(det)}
                for pred, det in zip(
                    predictions, crop_detections(predictions, cv2.imread(img_p))
                )
            ],
            "original_image": os.path.basename(img_p),
        },
    )
