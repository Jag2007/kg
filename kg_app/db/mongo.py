import os
from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover - dependency may be missing locally before install
    MongoClient = None


def _read_env_value(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is missing in {ENV_PATH}.")
    return value


def get_database():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

    mongo_uri = _read_env_value("MONGO_URI")
    db_name = _read_env_value("DB_NAME")

    if not mongo_uri:
        raise RuntimeError(f"MONGO_URI is missing in {ENV_PATH}.")
    if MongoClient is None:
        raise RuntimeError(
            "pymongo is not installed in the current environment. "
            "Run `pip install -r requirements.txt` after activating your virtual environment."
        )

    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=6000,
        appname="knowledge-graph-studio",
    )
    client.admin.command("ping")
    return client, client[db_name]
