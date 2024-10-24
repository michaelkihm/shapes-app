from contextlib import asynccontextmanager

from fastapi import FastAPI
from ultralytics import YOLO

ML_MODELS: dict[str, YOLO] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ML_MODELS["yolo"] = YOLO("yolo11n.pt")
    yield

    ML_MODELS.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}
