import os
import shutil
from typing import TypedDict

import cv2
import numpy as np
from fastapi import APIRouter, Depends, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results

from ..dependencies import get_yolo

templates = Jinja2Templates(directory="shapes/templates")
router = APIRouter()


class DetResult(TypedDict):
    conf: float
    box: list[int]  # box x,y,w,h
    label: str


class CropDetResult(DetResult):
    img: str


type TemplateDets = dict[str, list[CropDetResult]]


def imgp_from_index(i: int) -> str:
    return f"shapes/static/imgs/uploaded_image_{i}.jpg"


def convert_xywh_to_box(xywh_box: Tensor) -> list[int]:
    """
    Convert ultralytics xywh bounding box [x_center, y_center, width, height]
    to [x,y,w,h]
    """
    x_c, y_c, w, h = xywh_box[0].numpy().tolist()
    return [int(x_c - w / 2), int(y_c - h / 2), int(w), int(h)]


def extract_detections(results: list[Results], conf_thr=0.9) -> list[list[DetResult]]:
    predictions: list[list[DetResult]] = []
    names = results[0].names

    for result in results:
        img_redictions: list[DetResult] = []
        boxes = result.boxes
        for box in boxes:  # type:ignore
            conf = float(box.conf)
            if conf < conf_thr:
                continue
            img_redictions.append(
                {
                    "conf": round(conf, 2),
                    "box": convert_xywh_to_box(box.xywh),
                    "label": names[int(box.cls)],
                }
            )
        predictions.append(img_redictions)
    return predictions


def crop_detections(
    detections: list[DetResult], query_img: np.ndarray, img_i: int
) -> list[str]:
    paths: list[str] = []

    for i, det in enumerate(detections):
        x, y, w, h = det["box"]
        path = f"shapes/static/imgs/detection_{img_i}_{i}.jpg"
        paths.append(path)
        cv2.imwrite(
            path,
            query_img[y : y + h, x : x + w],
        )

    return paths


def build_template_detections(imgs_detections: list[list[DetResult]]) -> TemplateDets:
    result_dict: dict[str, list[CropDetResult]] = {}

    for i, image in enumerate(imgs_detections):
        src_img = imgp_from_index(i)
        result_dict[os.path.basename(src_img)] = [
            {**pred, "img": os.path.basename(det)}
            for pred, det in zip(image, crop_detections(image, cv2.imread(src_img), i))
        ]
    return result_dict


@router.post("/upload", response_class=HTMLResponse)
async def model_inference(
    request: Request, files: list[UploadFile], yolo: YOLO = Depends(get_yolo)
):
    # Download and save uploaded files
    for i, file in enumerate(files):
        img_p = imgp_from_index(i)
        with open(img_p, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    # Model inference
    results: list[Results] = yolo([imgp_from_index(i) for i in range(len(files))])
    detections = extract_detections(results)

    # Init and return HTML template
    return templates.TemplateResponse(
        "results.html",
        {"request": request, "detections": build_template_detections(detections)},
    )
