from fastapi import APIRouter, FastAPI, HTTPException
import json
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.fastapi_app.schemas.schemas import CategorySummary
from src.log.logger import logger
from src.fastapi_app.utils.config import config


base_router = APIRouter()
