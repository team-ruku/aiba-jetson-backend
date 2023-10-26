from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.middleware.cors import CORSMiddleware

from app.core import shutdown_event_handler, statup_event_handler

from yolo import YOLOStream

from vision.run import VisionDepth

from app.routes.v1 import v1_router

__version__ = "0.1.0"

app_config = {
    "title": "AIBA Backend",
    "description": "FastAPI Application for AIBA Project",
    "version": "0.1.0",
    "redoc_url": "/docs/redoc",
    "docs_url": "/docs/swagger",
}

app = FastAPI(**app_config)

app.add_event_handler("startup", statup_event_handler(app))
app.add_event_handler("shutdown", shutdown_event_handler(app))

app.yolo_instance = YOLOStream()
app.vision_instance = VisionDepth()

app.include_router(v1_router, prefix="/v1")


@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs/swagger")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
