from typing import Generator
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

engine: Engine

def get_db() -> Generator[Session, None, None]: ...
