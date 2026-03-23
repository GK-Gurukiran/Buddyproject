"""
Tests for chat endpoints.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models.database import async_engine, Base, AsyncSessionLocal, Tenant, User, UserRole
import bcrypt


@pytest.fixture(autouse=True)
async def setup_db():
    """Set up a fresh database for each test."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Seed test data
    async with AsyncSessionLocal() as db:
        tenant = Tenant(name="Test Org", slug="test-org", max_users=10)
        db.add(tenant)
        await db.flush()

        hashed = bcrypt.hashpw(b"testpass123", bcrypt.gensalt()).decode("utf-8")
        user = User(
            email="chatuser@example.com",
            username="chatuser",
            hashed_password=hashed,
            role=UserRole.USER,
            tenant_id=tenant.id,
        )
        db.add(user)
        await db.commit()

    yield

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def authed_client():
    """Create a client with auth token."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        login_res = await client.post("/api/v1/auth/login", json={
            "email": "chatuser@example.com",
            "password": "testpass123"
        })
        token = login_res.json()["access_token"]
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


@pytest.mark.asyncio
async def test_chat_requires_auth():
    """Test that chat endpoint requires authentication."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/api/v1/chat/", json={"message": "Hello"})
        assert response.status_code in [401, 403]


@pytest.mark.asyncio
async def test_list_sessions_empty(authed_client):
    """Test listing sessions when none exist."""
    response = await authed_client.get("/api/v1/chat/sessions")
    assert response.status_code == 200
    assert response.json() == []


@pytest.mark.asyncio
async def test_chat_creates_session(authed_client):
    """Test that sending a message creates a new session."""
    response = await authed_client.post("/api/v1/chat/", json={
        "message": "Hello, this is a test"
    })
    assert response.status_code == 200
    data = response.json()
    assert "session_id" in data
    assert "response" in data
    assert len(data["response"]) > 0


@pytest.mark.asyncio
async def test_chat_continue_session(authed_client):
    """Test continuing a conversation in existing session."""
    # First message
    res1 = await authed_client.post("/api/v1/chat/", json={
        "message": "What is 2+2?"
    })
    session_id = res1.json()["session_id"]

    # Second message in same session
    res2 = await authed_client.post("/api/v1/chat/", json={
        "message": "And what is 3+3?",
        "session_id": session_id
    })
    assert res2.status_code == 200
    assert res2.json()["session_id"] == session_id


@pytest.mark.asyncio
async def test_get_session_messages(authed_client):
    """Test retrieving messages from a session."""
    # Send a message
    chat_res = await authed_client.post("/api/v1/chat/", json={
        "message": "Test message for history"
    })
    session_id = chat_res.json()["session_id"]

    # Get messages
    response = await authed_client.get(f"/api/v1/chat/sessions/{session_id}/messages")
    assert response.status_code == 200
    messages = response.json()
    assert len(messages) >= 2  # user + assistant
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
