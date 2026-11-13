"""Database models (SQLAlchemy 2.0). SQLite locally; set DATABASE_URL for PostgreSQL later."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import settings


def now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class Project(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200))
    original_name: Mapped[str] = mapped_column(String(300), default="")
    media_file: Mapped[str] = mapped_column(String(100))          # safe internal file name
    media_info: Mapped[str] = mapped_column(Text, default="{}")   # JSON from ffprobe
    language: Mapped[str] = mapped_column(String(10), default="auto")
    words: Mapped[str] = mapped_column(Text, default="[]")        # JSON list of word/phrase timings
    style: Mapped[str] = mapped_column(Text, default="{}")        # JSON caption style
    warnings: Mapped[str] = mapped_column(Text, default="[]")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    jobs: Mapped[list["Job"]] = relationship(back_populates="project", cascade="all, delete-orphan")
    exports: Mapped[list["Export"]] = relationship(back_populates="project", cascade="all, delete-orphan")

    def get_info(self) -> dict:
        return json.loads(self.media_info or "{}")

    def get_words(self) -> list[dict]:
        return json.loads(self.words or "[]")

    def get_style(self) -> dict:
        return json.loads(self.style or "{}")


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20))       # transcribe | export
    status: Mapped[str] = mapped_column(String(20), default="queued")  # queued running completed failed cancelled
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    message: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    where: Mapped[str] = mapped_column(String(10), default="local")   # local (never cloud in this version)
    params: Mapped[str] = mapped_column(Text, default="{}")
    result: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    project: Mapped[Project] = relationship(back_populates="jobs")


class Export(Base):
    __tablename__ = "exports"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    job_id: Mapped[str] = mapped_column(String(32), default="")
    folder: Mapped[str] = mapped_column(String(100))
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    options: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    project: Mapped[Project] = relationship(back_populates="exports")


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, future=True)

if settings.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_conn, _):  # foreign keys + safer concurrent access
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db() -> None:
    """Create tables that do not exist yet. Never drops or resets anything."""
    Base.metadata.create_all(engine)


def get_session():
    with SessionLocal() as session:
        yield session
