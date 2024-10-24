from ultralytics import YOLO

ML_MODELS: dict[str, YOLO] = {}


def get_yolo() -> YOLO:
    if "yolo" not in ML_MODELS:
        raise KeyError("Model yolo was not loaded")
    return ML_MODELS["yolo"]
