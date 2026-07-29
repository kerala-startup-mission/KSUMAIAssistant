# KSUM AI Voice Assistant

An optimized, fully offline-capable AI Voice Chatbot designed for the Kerala Startup Mission (KSUM). This system utilizes a Retrieval-Augmented Generation (RAG) architecture to provide accurate, context-aware answers regarding KSUM's schemes, grants, and startup processes. 

The project has been architected as a Dockerized Microservice, featuring hyper-natural Text-to-Speech (TTS), local Speech-to-Text (STT), advanced Two-Stage Document Retrieval, and a self-healing vector database environment.

## Architecture & Tech Stack

*  Deployment & Orchestration:              Docker & Docker Compose
*  LLM Engine:                              gemma4:e4b (Running natively on host via Ollama for hardware acceleration)
*  Stage 1 Retrieval (Bi-Encoder):          thenlper/gte-small (Offline/Cached)
*  Stage 2 Re-Ranking (Cross-Encoder):      ms-marco-MiniLM-L-6-v2 (Offline/Cached)
*  Vector Database:                         ChromaDB (Containerized)
*  Data Pipeline:                           crawl4ai (Web Scraping) & Gemma (Chunk Summarization)
*  Backend API:                             FastAPI (Python) with slowapi Rate Limiting
*  Voice Modules:                           Edge TTS (Speech Synthesis) & Faster-Whisper (Speech Recognition)
*  Frontend:                                HTML5, Tailwind CSS, JavaScript (MediaRecorder API)

---

## Project Structure

The codebase is organized into distinct, isolated microservices to ensure scalability and easy maintenance:

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
├── .dockerignore              # Prevents container bloat by ignoring venv, cache, and old builds
├── docker-compose.yml         # Orchestrates the API and ChromaDB containers with network isolation
├── Dockerfile                 # Image blueprint for the Python FastAPI backend
└── requirements.txt           # Python dependencies

---

## Commands

* To run docker normally:          docker compose up -d
* To run the scraper:              python -m pipeline.corescrape
* To run the ingestion             python -m pipeline.ingest
* To run the frontend:             python -m http.server 5500 -d frontend
* Website:                         http://localhost:5500
* Docker Clean Wipe:               docker compose down -v --rmi all
* Docker Deep clean:               docker system prune -a --volumes -f
* Docker clear builder cache:      docker builder prune -a -f
* To rebuild the Docker stack:     docker compose up --build -d

---