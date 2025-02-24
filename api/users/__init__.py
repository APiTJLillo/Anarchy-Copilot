"""Users API package."""
from fastapi import APIRouter

router = APIRouter(
    tags=["users"]  # Remove prefix since it's handled in main.py
)

from .endpoints import *  # noqa
