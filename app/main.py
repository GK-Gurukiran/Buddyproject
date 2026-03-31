"""
FastAPI application entry point.
Scalable AI Chatbot Platform with multi-agent orchestration,
multi-tenant architecture, cross-chat memory, and voice capabilities.
"""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models.database import init_db
from app.routers import auth, tenants, chat, memory
from app.voice import router as voice

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    await init_db()
    logger.info("Database initialized")

    # Seed a default tenant if none exists (for development)
    if settings.debug:
        await _seed_default_data()

    yield

    # Shutdown
    logger.info("Shutting down...")


async def _seed_default_data():
    """Create default tenant and admin user for development."""
    from sqlalchemy import select
    import bcrypt
    from app.models.database import AsyncSessionLocal, Tenant, User, UserRole

    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    async with AsyncSessionLocal() as db:
        # Check if default tenant exists
        result = await db.execute(select(Tenant).where(Tenant.slug == "default"))
        if result.scalar_one_or_none():
            return

        # Create default tenant
        tenant = Tenant(name="Default Organization", slug="default", max_users=100)
        db.add(tenant)
        await db.flush()

        # Create admin user
        admin = User(
            email="admin@example.com",
            username="admin",
            hashed_password=hash_password("admin123"),
            role=UserRole.SUPER_ADMIN,
            tenant_id=tenant.id,
        )
        db.add(admin)
        await db.commit()
        logger.info("Default tenant and admin user created (admin@example.com / admin123)")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "A scalable AI chatbot platform featuring multi-agent orchestration (LangGraph), "
        "multi-tenant vector storage (Qdrant), cross-chat memory (Mem0), "
        "and real-time voice capabilities (LiveKit)."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    swagger_ui_parameters={"tryItOutEnabled": True},
)

# Custom docs page that works offline (fallback if CDN fails)
from fastapi.responses import HTMLResponse

@app.get("/api-docs", response_class=HTMLResponse, include_in_schema=False)
async def custom_docs():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{settings.app_name} - API Docs</title>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui.css" />
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist@5.11.0/swagger-ui-bundle.js"></script>
        <script>
        SwaggerUIBundle({{
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset],
            layout: "BaseLayout",
            tryItOutEnabled: true,
        }})
        </script>
    </body>
    </html>
    """

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1")
app.include_router(tenants.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(memory.router, prefix="/api/v1")
app.include_router(voice.router, prefix="/api/v1")


# Serve frontend UI
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    from fastapi.responses import FileResponse

    @app.get("/", include_in_schema=False)
    async def serve_frontend():
        """Serve the chat UI."""
        return FileResponse(os.path.join(_static_dir, "index.html"))

    app.mount("/static", StaticFiles(directory=_static_dir), name="static")
else:
    @app.get("/", tags=["Health"])
    async def root():
        """Health check endpoint."""
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "status": "running",
        }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check with live status of all services."""
    from app.memory.vector_store import vector_store
    from app.voice.livekit_agent import livekit_manager

    # Check search tool status
    search_status = "tavily" if settings.tavily_api_key else "duckduckgo (free fallback)"
    scrape_status = "firecrawl" if settings.firecrawl_api_key else "httpx+beautifulsoup (free fallback)"

    return {
        "status": "healthy",
        "version": settings.app_version,
        "llm_provider": settings.llm_provider,
        "services": {
            "database": "connected",
            "llm": f"{settings.llm_provider} ({settings.openrouter_model})" if settings.llm_provider == "openrouter" else settings.llm_provider,
            "vector_store": "in-memory fallback" if vector_store.is_using_fallback else "qdrant",
            "memory": "mem0 + qdrant" if not vector_store.is_using_fallback else "in-memory fallback",
            "web_search": search_status,
            "web_scraper": scrape_status,
            "voice": "livekit ready" if livekit_manager.is_available else "not available (start docker-compose up livekit)",
            "mcp_server": "available (Python 3.10+ required)" if settings.debug else "available",
        },
    }
