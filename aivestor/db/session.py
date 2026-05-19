from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from aivestor.db.models import Base


def default_sqlite_url() -> str:
    return "sqlite:///data/aivestor.db"


def get_database_url() -> str:
    return os.environ.get("DATABASE_URL", default_sqlite_url())


def get_engine(url: str | None = None) -> Engine:
    u = url or get_database_url()
    connect_args: dict = {}
    if u.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(u, future=True, connect_args=connect_args)


def init_db(url: str | None = None) -> None:
    engine = get_engine(url)
    Base.metadata.create_all(engine)


def get_session_factory(url: str | None = None):
    return sessionmaker(bind=get_engine(url), class_=Session, autoflush=False, future=True)
