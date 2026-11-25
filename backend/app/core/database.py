import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://codegraph:codegraph_dev_password@localhost:5433/codegraph_db",
)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class CodeNode(Base):
    __tablename__ = "code_nodes"

    id = Column(Integer, primary_key=True, index=True)
    repo_url = Column(String, index=True, nullable=False)
    repo_name = Column(String, index=True, nullable=False)
    file_path = Column(String, nullable=False)
    language = Column(String, nullable=False)
    node_type = Column(String, nullable=False)
    symbol_name = Column(String, index=True, nullable=False)
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    code = Column(Text, nullable=False)
    calls = Column(JSON, default=list)
    embedding = Column(Vector(384))
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db() -> None:
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)
