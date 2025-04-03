"""Recon API module."""

from fastapi import APIRouter

router = APIRouter(
    prefix="/api/recon",
    tags=["recon"],
    responses={404: {"description": "Not found"}},
)

from . import endpoints  # noqa
