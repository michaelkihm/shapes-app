from ultralytics import YOLO

from .main import ML_MODELS


def get_yolo() -> YOLO:
    if "yolo" not in ML_MODELS:
        raise KeyError("Model yolo was not loaded")
    return ML_MODELS["yolo"]
