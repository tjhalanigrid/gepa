# System Architecture Documentation

This document describes the high-level architecture of the **Car Damage MVP** system.

## 🏗️ System Flow & Architecture Diagram

```mermaid
graph TD
    A["Raw Claim Image(s) Uploaded"] --> B["API Gateway (backend/app/routers/assessment.py)"]
    B --> C["Pipeline Orchestrator (pipeline/orchestrator.py)"]
    
    subgraph "Machine Learning Model Stage"
        C --> D["Damage Detection (infer.py)"]
        C --> E["Part Segmentation (infer.py)"]
        C --> F["Plate & Recurrence Detection (infer.py)"]
    end
    
    D --> G["Context Builder (pipeline/context_builder.py)"]
    E --> G
    F --> G
    
    G --> H["VLM Reasoner Client (models/vlm_reasoning/vlm_client.py)"]
    
    subgraph "Visual Language Model Analysis"
        H --> I["Local Ollama (Qwen2.5-VL / Gemma4)"]
    end
    
    I --> J["Validated Schema Contract (pipeline/schema.py)"]
    J --> K["Cost Estimator Engine (experiments/estimator.py)"]
    K --> L["Relational Database Persistence (PostgreSQL / MongoDB)"]
    L --> M["Interactive Executive Claims Dashboard (dashboard/app.py)"]
```

## 📂 Modular Design Philosophy
Each component is fully encapsulated to ensure high development isolation:
- **`models/`**: Submodules representing independent neural network boundaries. Other layers MUST ONLY communicate with models via their public `infer.py` interfaces.
- **`pipeline/`**: The orchestration plane that runs models sequentially and enforces validated output schemas using Pydantic.
- **`shared/`**: Common cross-module utility functions (Base64 encoding/decoding, claims logging formats).
- **`backend/`**: Relational FastAPI service layer handling claims persistence.
- **`dashboard/`**: Streamlit claims dashboard.
