import logging
from typing import Any, Callable, Coroutine, Dict

from fastapi.routing import APIRoute
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class LoggingMiddleware(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        origin_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            response_from_origin: Response = await origin_route_handler(request)
            return response_from_origin

        return custom_route_handler

    @staticmethod
    def __check_json_body(request: Request) -> bool:
        if (
            request.method in ("POST", "PUT", "PATCH")
            and request.headers.get("content-type") == "application/json"
        ):
            return True
        return False

    async def _request_logger(self, request: Request) -> None:
        extra_information: Dict[str, Any] = {
            "request_method": request.method,
            "origin_url": request.url.path,
            "headers": request.headers,
            "query_params": request.query_params,
        }

        if self.__check_json_body(request):
            request_body = await request.body()
            extra_information["body"] = request_body.decode("UTF-8")

        logger.info("request", extra=extra_information)

    @staticmethod
    def _response_logger(request: Request, response: Response) -> None:
        extra_information: Dict[str, str] = {
            "request_method": request.method,
            "origin_url": request.url.path,
            "body": response.body.decode("UTF-8"),
        }

        logger.info("response", extra=extra_information)
