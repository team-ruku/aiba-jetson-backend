from fastapi import APIRouter

from app.middlewares import LoggingMiddleware

from .heartbeat import heartbeat_router

v1_router = APIRouter(route_class=LoggingMiddleware)

v1_router.include_router(heartbeat_router, prefix="/health", tags=["health"])
