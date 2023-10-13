from fastapi import APIRouter

from app.middlewares import LoggingMiddleware

v1_router = APIRouter(route_class=LoggingMiddleware)
