import os
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", "08fe8c5a-e092-4303-8f1a-6c3119d16c36")

DB_USER     = "postgres"
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST     = "127.0.0.1"
DB_PORT     = 5432
DB_NAME     = "ENTSOE-E"

COUNTRY_CODE = "DE_LU"
LOOKBACK_DAYS = 100

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_engine():
    return create_engine(
        URL.create(
            drivername="postgresql+psycopg2",
            username=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
        )
    )
