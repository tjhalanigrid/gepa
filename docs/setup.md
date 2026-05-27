# Setup & Installation Guide

This document describes how to configure your environment and run the modular MVP.

## 🐍 Prerequisites
- **Python**: version `3.11`
- **Docker & Docker Compose** (for PostgreSQL database vectors and MongoDB services)
- **Local Ollama** (running `qwen2.5vl:7b` and `gemma4:latest` models)

---

## 🛠️ Step-by-Step Local Environment Setup

### 1. Build and Run Databases (Docker Compose)
Launch standard Postgres and MongoDB vector database structures using:
```bash
docker compose -f docker/docker-compose.yml up -d
```
*This automatically initializes your tables using the schemas stored in [schema.sql](file:///Users/kokumar/Desktop/vehicle-damage-ai/backend/migrations/schema.sql).*

### 2. Download local Ollama VLM Models
Ensure your local Ollama instance has the correct vision models downloaded:
```bash
ollama pull qwen2.5vl:7b
ollama pull gemma4:latest
```

### 3. Install Python Dependencies
Install pinned shared base requirements and FastAPI packages:
```bash
pip install -r requirements-base.txt
pip install -r backend/requirements.txt
pip install -r dashboard/requirements.txt
```
