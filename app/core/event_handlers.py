from fastapi import FastAPI


def statup_event_handler(app: FastAPI):
    def on_startup():
        pass  # do loggin shit

    return on_startup
