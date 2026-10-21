from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from nl_energy_pipeline.config import settings


def get_engine(url: str | None = None) -> Engine:
    # Pin the session time zone so timestamptz values always come back in UTC.
    return create_engine(
        url or settings.database_url,
        pool_pre_ping=True,
        connect_args={"options": "-c timezone=utc"},
    )


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Commit on success, roll back on any error."""
    session = sessionmaker(engine, expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
