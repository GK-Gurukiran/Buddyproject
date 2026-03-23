"""
Shared test configuration and fixtures.
"""

import os
import pytest

# Use in-memory SQLite for tests
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["SECRET_KEY"] = "test-secret-key"
os.environ["DEBUG"] = "true"


@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"
