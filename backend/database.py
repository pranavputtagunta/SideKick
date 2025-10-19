from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# This is the connection string for your local PostgreSQL database running in Docker.
# Format: "postgresql://<user>:<password>@<host>/<database_name>"
SQLALCHEMY_DATABASE_URL = "postgresql://sidekick:sidekick@localhost/sidekick"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Each instance of SessionLocal will be a database session.
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# This Base class will be used by our models to inherit from.
Base = declarative_base()
