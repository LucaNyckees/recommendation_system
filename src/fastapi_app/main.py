from fastapi import FastAPI
from src.fastapi_app.routes.base import base_router


app = FastAPI()
app.include_router(base_router)


@app.get("/")
def read_root():
    return {"message": r"Go to https://{host}:{port}/docs to get the complete set of routes."}
