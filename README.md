# Scalable AI Chatbot Platform

A scalable AI chatbot platform built with **FastAPI** and **LangGraph**, featuring multi-agent orchestration, multi-tenant vector storage, cross-chat memory, and voice call capabilities via **LiveKit**.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      FastAPI Backend                         │
│  ┌─────────┐  ┌─────────┐  ┌────────┐  ┌──────┐  ┌──────┐  │
│  │  Auth   │  │  Chat   │  │ Memory │  │Voice │  │Tenant│  │
│  │ Router  │  │ Router  │  │ Router │  │Router│  │Router│  │
│  └────┬────┘  └────┬────┘  └───┬────┘  └──┬───┘  └──┬───┘  │
│       │            │           │           │         │       │
│  ┌────┴────────────┴───────────┴───────────┴─────────┴───┐  │
│  │              JWT Auth + RBAC Middleware                │  │
│  └───────────────────────┬───────────────────────────────┘  │
│                          │                                   │
│  ┌───────────────────────┴───────────────────────────────┐  │
│  │             LangGraph Multi-Agent System               │  │
│  │  ┌────────────┐  ┌──────────┐  ┌──────────────────┐  │  │
│  │  │ Supervisor │──│ Research │  │  Response         │  │  │
│  │  │   Agent    │  │  Agent   │──│  Assembler        │  │  │
│  │  │            │──│          │  │                    │  │  │
│  │  │  (Router)  │  │ (Tavily) │  │ (Combines results)│  │  │
│  │  │            │  └──────────┘  └──────────────────┘  │  │
│  │  │            │  ┌──────────┐                         │  │
│  │  │            │──│ Scraper  │                         │  │
│  │  └────────────┘  │  Agent   │                         │  │
│  │                   │(Firecrawl)                         │  │
│  │                   └──────────┘                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│  ┌───────────┐  ┌───────┴──────┐  ┌──────────┐             │
│  │  Mem0     │  │    Qdrant    │  │ LiveKit  │             │
│  │ (Memory)  │  │(Vector Store)│  │ (Voice)  │             │
│  └───────────┘  └──────────────┘  └──────────┘             │
└──────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component          | Technology     |
|--------------------|---------------|
| Backend API        | FastAPI       |
| Agent Orchestration| LangGraph     |
| Memory Layer       | Mem0          |
| Vector Store       | Qdrant        |
| Voice              | LiveKit       |
| Web Search         | Tavily        |
| Web Scraping       | Firecrawl     |
| Tool Protocol      | MCP           |
| LLM                | OpenAI GPT-4o |
| Auth               | JWT + RBAC    |

## Features

- **Multi-Agent System** — Supervisor routes queries to Research and Scraper agents
- **Multi-Tenant Architecture** — Complete data isolation between organizations
- **Role-Based Access Control** — Super Admin, Tenant Admin, User, Viewer roles
- **Cross-Chat Memory** — Remembers context from previous conversations (Mem0 + Qdrant)
- **Real-Time Voice** — Talk to the AI agent via LiveKit (VAD, STT, TTS)
- **MCP Server** — Expose tools via Model Context Protocol standard
- **Docker Ready** — One command to spin up everything

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# 1. Clone and enter the project
cd scalable-ai-chatbot

# 2. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys

# 3. Start everything
docker compose up --build

# API is now running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env
# Edit .env with your API keys

# 4. Start Qdrant (requires Docker)
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 5. Run the API server
uvicorn app.main:app --reload --port 8000

# 6. (Optional) Run voice worker in separate terminal
python -m app.voice.worker
```

## API Endpoints

### Authentication
| Method | Endpoint               | Description         |
|--------|------------------------|---------------------|
| POST   | `/api/v1/auth/register`| Register a new user |
| POST   | `/api/v1/auth/login`   | Login, get JWT token|
| GET    | `/api/v1/auth/me`      | Get current user    |
| GET    | `/api/v1/auth/users`   | List users (admin)  |

### Chat
| Method | Endpoint                                  | Description              |
|--------|-------------------------------------------|--------------------------|
| POST   | `/api/v1/chat/`                           | Send a message           |
| GET    | `/api/v1/chat/sessions`                   | List chat sessions       |
| GET    | `/api/v1/chat/sessions/{id}/messages`     | Get session messages     |
| DELETE | `/api/v1/chat/sessions/{id}`              | Delete a session         |

### Memory
| Method | Endpoint                    | Description          |
|--------|-----------------------------|----------------------|
| POST   | `/api/v1/memory/search`     | Search memories      |
| GET    | `/api/v1/memory/`           | List all memories    |
| DELETE | `/api/v1/memory/{id}`       | Delete a memory      |
| DELETE | `/api/v1/memory/`           | Clear all memories   |

### Voice
| Method | Endpoint                        | Description           |
|--------|---------------------------------|-----------------------|
| POST   | `/api/v1/voice/token`           | Get voice room token  |
| GET    | `/api/v1/voice/rooms`           | List active rooms     |
| DELETE | `/api/v1/voice/rooms/{name}`    | End voice session     |

### Tenants (Super Admin)
| Method | Endpoint                    | Description         |
|--------|-----------------------------|---------------------|
| POST   | `/api/v1/tenants/`          | Create tenant       |
| GET    | `/api/v1/tenants/`          | List all tenants    |
| GET    | `/api/v1/tenants/{id}`      | Get tenant details  |
| DELETE | `/api/v1/tenants/{id}`      | Deactivate tenant   |

## Usage Example

```bash
# 1. Login (default dev credentials)
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","password":"admin123"}' \
  | jq -r '.access_token')

# 2. Send a chat message
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"message": "What are the latest trends in AI?"}'

# 3. Search your memories
curl -X POST http://localhost:8000/api/v1/memory/search \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "AI trends", "limit": 5}'
```

## Required API Keys

| Service    | Get Key From                                  | Required |
|------------|-----------------------------------------------|----------|
| OpenAI     | https://platform.openai.com/api-keys          | Yes      |
| Tavily     | https://app.tavily.com/                        | Yes      |
| Firecrawl  | https://www.firecrawl.dev/                     | Yes      |
| LiveKit    | https://cloud.livekit.io/                      | For voice|
| Qdrant     | Self-hosted or https://cloud.qdrant.io/        | Yes      |

## MCP Server

The platform exposes an MCP server for external LLM clients:

```bash
# Run the MCP server (stdio transport)
python -m app.mcp_server.server
```

Available MCP tools: `web_search`, `scrape_url`, `search_and_scrape`

## Project Structure

```
scalable-ai-chatbot/
├── app/
│   ├── main.py                  # FastAPI entry point
│   ├── config.py                # Settings from .env
│   ├── agents/
│   │   ├── graph.py             # LangGraph workflow
│   │   ├── state.py             # Shared agent state
│   │   ├── supervisor.py        # Supervisor (router) agent
│   │   ├── research_agent.py    # Web research agent
│   │   ├── scraper_agent.py     # Web scraper agent
│   │   └── tools.py             # Tool definitions
│   ├── auth/
│   │   ├── jwt_handler.py       # JWT creation/verification
│   │   ├── rbac.py              # Role-based access control
│   │   └── dependencies.py      # FastAPI auth dependencies
│   ├── memory/
│   │   ├── mem0_client.py       # Cross-chat memory (Mem0)
│   │   └── vector_store.py      # Qdrant vector store
│   ├── voice/
│   │   ├── livekit_agent.py     # LiveKit room management
│   │   ├── pipeline.py          # Voice AI pipeline
│   │   └── worker.py            # LiveKit agent worker
│   ├── mcp_server/
│   │   ├── server.py            # MCP server
│   │   └── tools.py             # MCP tool handlers
│   ├── routers/
│   │   ├── auth.py              # Auth endpoints
│   │   ├── chat.py              # Chat endpoints
│   │   ├── memory.py            # Memory endpoints
│   │   ├── voice.py             # Voice endpoints
│   │   └── tenants.py           # Tenant management
│   ├── models/
│   │   ├── database.py          # SQLAlchemy models
│   │   └── schemas.py           # Pydantic schemas
│   └── tenants/
│       └── manager.py           # Tenant operations
├── tests/
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT
