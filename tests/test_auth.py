"""
Tests for authentication and authorization.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.models.database import init_db, async_engine, Base


@pytest.fixture(autouse=True)
async def setup_db():
    """Set up a fresh database for each test."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def seeded_tenant(client: AsyncClient):
    """Seed a default tenant for testing."""
    from app.models.database import AsyncSessionLocal, Tenant
    async with AsyncSessionLocal() as db:
        tenant = Tenant(name="Test Org", slug="test-org", max_users=10)
        db.add(tenant)
        await db.commit()
        await db.refresh(tenant)
        return tenant


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Test that the health endpoint returns OK."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_register_and_login(client: AsyncClient, seeded_tenant):
    """Test user registration followed by login."""
    # Register
    reg_response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": "testuser@example.com",
            "username": "testuser",
            "password": "securepass123",
            "tenant_slug": "test-org",
        },
    )
    assert reg_response.status_code == 201
    user_data = reg_response.json()
    assert user_data["email"] == "testuser@example.com"

    # Login
    login_response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "testuser@example.com",
            "password": "securepass123",
        },
    )
    assert login_response.status_code == 200
    token_data = login_response.json()
    assert "access_token" in token_data


@pytest.mark.asyncio
async def test_login_invalid_credentials(client: AsyncClient):
    """Test login with wrong credentials returns 401."""
    response = await client.post(
        "/api/v1/auth/login",
        json={
            "email": "nonexistent@example.com",
            "password": "wrongpassword",
        },
    )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_protected_endpoint_without_token(client: AsyncClient):
    """Test that protected endpoints reject unauthenticated requests."""
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 403  # No credentials
