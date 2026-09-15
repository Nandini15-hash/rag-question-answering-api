import os
import shutil
import tempfile
from pathlib import Path

import pytest

TEST_DIR = Path(tempfile.mkdtemp(prefix="rag_test_"))
os.environ["EMBEDDING_PROVIDER"] = "offline"  # zero-network, deterministic, fast
os.environ["UPLOAD_DIR"] = str(TEST_DIR / "uploads")
os.environ["INDEX_DIR"] = str(TEST_DIR / "index")
os.environ["JOB_DB_PATH"] = str(TEST_DIR / "jobs.db")
os.environ["METRICS_LOG_PATH"] = str(TEST_DIR / "metrics.jsonl")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session", autouse=True)
def cleanup():
    yield
    shutil.rmtree(TEST_DIR, ignore_errors=True)
