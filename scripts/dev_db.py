"""Start a throwaway local Postgres (no Docker needed) and print its SQLAlchemy URL."""

import pgserver

server = pgserver.get_server(".pgdata", cleanup_mode=None)  # keep running after this script exits
server.psql("SELECT 1 FROM pg_database WHERE datname='nl_energy'")
if "(0 rows)" in server.psql("SELECT 1 FROM pg_database WHERE datname='nl_energy'"):
    server.psql("CREATE DATABASE nl_energy")

host = server.get_uri().split("host=")[1]
print(f"postgresql+psycopg://postgres:@/nl_energy?host={host}")
