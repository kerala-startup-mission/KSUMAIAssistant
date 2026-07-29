# KSUM AI Voice Assistant

An optimized, fully offline-capable AI Voice Chatbot designed for the Kerala Startup Mission (KSUM). This system utilizes a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers regarding KSUM's schemes, grants, and startup processes. 

The project has been architected as a Dockerized Microservice, featuring hyper-natural Text-to-Speech (TTS), local Speech-to-Text (STT), advanced Two-Stage Document Retrieval, and a self-healing vector database environment.

---
## Architecture & Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Deployment** | Docker & Compose | Containerized orchestration and network isolation. |
| **LLM Engine** | `gemma4:e4b` | Runs natively on host via Ollama for hardware acceleration. |
| **Bi-Encoder** | `thenlper/gte-small` | Stage 1 Retrieval (Offline/Cached). |
| **Cross-Encoder** | `ms-marco-MiniLM-L-6-v2`| Stage 2 Re-Ranking (Offline/Cached). |
| **Vector DB** | ChromaDB | Containerized, self-healing vector database. |
| **Backend API** | FastAPI (Python) | Handles Background Tasks, Chat, STT, and Admin endpoints. |
| **Voice Modules** | Edge TTS & Whisper | In-memory text-to-speech and local speech recognition. |
| **Frontend UI** | HTML5 / Tailwind | Modern glassmorphic interfaces for both Users and Admins. |

---

## Security Features

* **Zero-Trust XML Sandbox:** User inputs are wrapped in strict XML boundary tags (`<user_input>`) during RAG generation. The system prompt is engineered to reject all prompt injection attacks, roleplay overrides, and "Developer Mode" exploits.
* **Rate Limiting:** API endpoints are protected against spam and DDoS (e.g., 10 chats/min, 20 uploads/min) using `slowapi`.
* **Network Isolation:** ChromaDB runs on an internal Docker network bridge, completely walled off from public internet access.

---

## Project Structure

The codebase is organized into distinct, isolated microservices to ensure scalability and easy maintenance:

```text
ksum - gemma - docker/
│
├── backend/                   # The Core AI API Service
│   ├── app.py                 # FastAPI server (Handles Chat, TTS, STT, and Rate Limiting endpoints)
│   ├── rag_engine.py          # Two-Stage RAG logic, query contextualization, and vector retrieval
│   └── config.py              # Centralized configuration (Models, hostnames, chunk sizes)
│
├── pipeline/                  # Automated Data Ingestion
│   ├── corescrape.py          # Scrapes KSUM URLs and formats them into RAG-friendly markdown
│   └── ingest.py              # Smart-Append script: Summarizes chunks, computes embeddings, and populates ChromaDB
│
├── frontend/                  # User Interface
│   │── index.html             # Modern, responsive web interface connecting to the API
│   └── admin.html             # Admin Dashboard for file uploads, sync, and live logs
│
├── data/                      # Raw Knowledge Base
│   └── core_website_data/     # Local storage for scraped markdown files 
│
├── .gitignore                 # Prevents pushing sensitive data and vector DB files
├── .dockerignore              # Prevents container bloat by ignoring venv, cache, and old builds
├── docker-compose.yml         # Orchestrates the API and ChromaDB containers with network isolation
├── Dockerfile                 # Image blueprint for the Python FastAPI backend
└── requirements.txt           # Python dependencies
```

---
## Getting Started (Commands)

### Core Application
Start the entire stack in the background:
```bash
docker compose up -d
```

Serve the frontend interfaces (run from the project root):
```bash
python -m http.server 5500 -d frontend
```

* **User Chat Website:** `http://localhost:5500/index.html`
* **Admin Control Panel:** `http://localhost:5500/admin.html`

### Maintenance & Rebuilding
Rebuild the Docker stack after making code changes:
```bash
docker compose up --build -d
```

Clean wipe all containers and data volumes:
```bash
docker compose down -v --rmi all
```

Deep clean Docker cache (use if builds are failing):
```bash
docker system prune -a --volumes -f
docker builder prune -a -f
```
---