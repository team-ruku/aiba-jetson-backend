from fastapi import APIRouter

stream_router = APIRouter()


@stream_router.get("/")
def get_hearbeat():
    pass
