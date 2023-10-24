from typing import Callable

from fastapi import FastAPI
from loguru import logger


def statup_event_handler(app: FastAPI) -> Callable:
    def on_startup() -> None:
        logger.level("YOLO", no=38, color="<yellow>", icon="🐍")
        logger.info("AIBA backend instance has been started.")

        app.yolo_instance.setup()

    return on_startup


def shutdown_event_handler(app: FastAPI) -> Callable:
    def on_shutdown() -> None:
        logger.info("AIBA backend instance has been stopped.")

        app.yolo_instance.end_instance()

    return on_shutdown
