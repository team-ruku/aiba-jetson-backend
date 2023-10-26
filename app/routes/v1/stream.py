from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

stream_router = APIRouter()


@stream_router.get("/yolo")
def stream_yolo(request: Request):
    return StreamingResponse(
        request.app.yolo_instance.main(args="YOLO"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@stream_router.get("/tdoa")
def stream_yolo(request: Request):
    return StreamingResponse(
        request.app.yolo_instance.main(args="TDOA"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
