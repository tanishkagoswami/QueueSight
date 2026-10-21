"""Alembic runs this for every migration command: it connects to the database and applies the steps in versions/."""
from logging.config import fileConfig

from alembic import context
from app.config import DATABASE_URL
from app.db import SQLModel, is_ours
from sqlalchemy import create_engine, pool

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = SQLModel.metadata    # our tables, so `alembic revision --autogenerate` can spot model changes

# The database to migrate: the one a test hands in, otherwise DATABASE_URL (the same one the app uses)
url = config.get_main_option("sqlalchemy.url") or DATABASE_URL
if not url:
    raise RuntimeError("DATABASE_URL is not set, e.g. "
                       "$env:DATABASE_URL=\"postgresql+psycopg://queuesight:queuesight@localhost:5433/queuesight\"")


def run_migrations_offline() -> None:
    """`alembic upgrade head --sql`: print the SQL instead of running it."""
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"}, include_object=is_ours)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect and run every step that is still missing, all inside one transaction."""
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata,
                          include_object=is_ours, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()