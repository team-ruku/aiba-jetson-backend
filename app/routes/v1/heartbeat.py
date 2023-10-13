from fastapi import APIRouter

from app.models import HearbeatResult

heartbeat_router = APIRouter()


@heartbeat_router.get("/", response_model=HearbeatResult, name="heartbeat")
def get_hearbeat() -> HearbeatResult:
    heartbeat = HearbeatResult(is_alive=True)
    return heartbeat
