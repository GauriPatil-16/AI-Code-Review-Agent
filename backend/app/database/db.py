from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import BACKEND_ROOT, get_settings


class Base(DeclarativeBase):
    pass


def database_url() -> str:
    url = get_settings().database_url
    if url.startswith("sqlite:///") and ":memory:" not in url:
        path = Path(url.removeprefix("sqlite:///"))
        if not path.is_absolute():
            path = BACKEND_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return "sqlite:///" + path.as_posix()
    return url


engine = create_engine(database_url(), connect_args={"check_same_thread": False, "timeout": 30})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db():
    from app.database import models  # noqa: F401

    Base.metadata.create_all(engine)


def get_db():
    with SessionLocal() as session:
        yield session
