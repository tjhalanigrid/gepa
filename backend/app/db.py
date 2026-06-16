"""
Database layer — SQLAlchemy engine, session factory, and declarative base.

Persists user accounts, vehicles, claims, and per-user settings in PostgreSQL.
Connection string comes from the DATABASE_URL env var, defaulting to a local
trust-auth Postgres instance (matching the dev machine).

Routes are defined as sync `def` functions, so FastAPI runs them in a threadpool
and a plain synchronous SQLAlchemy session is safe and simple here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Load .env from repo root (two levels up from this file: backend/app/db.py)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://localhost:5432/veh_dmg_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a session and always closes it."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables if they don't yet exist. Called on app startup."""
    # Import models so they register on Base.metadata before create_all.
    from . import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
