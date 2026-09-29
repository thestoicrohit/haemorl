"""Shared test setup: run the app against a throwaway data directory."""
import os
import sys
import tempfile

import pytest

# Must happen before `app` is imported: the app reads DATA_DIR at import time,
# and every save goes there — so tests never touch the real data/haemorl_db.json.
os.environ["DATA_DIR"] = tempfile.mkdtemp(prefix="haemorl-test-")
for var in ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"):
    os.environ.pop(var, None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402
import app as app_module  # noqa: E402


@pytest.fixture(scope="session")
def client():
    # The context manager runs startup, which seeds the in-memory DB.
    with TestClient(app_module.app) as c:
        yield c
