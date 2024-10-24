from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from .dependencies import ML_MODELS
from .routers import index, uploads


@asynccontextmanager
async def lifespan(app: FastAPI):
    ML_MODELS["yolo"] = YOLO("yolo11n.pt")
    yield

    ML_MODELS.clear()


app = FastAPI(lifespan=lifespan)

# Routers
app.include_router(index.router)
app.include_router(uploads.router)


# Enable CORS if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
