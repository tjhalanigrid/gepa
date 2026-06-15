# Vehicle Damage Assessment — backend API image.
#
# IMPORTANT: build from the REPOSITORY ROOT (not from backend/), because the API
# imports pipeline/, models/, and shared/:
#
#     docker build -f backend/Dockerfile -t veh-dmg-backend .
#     docker run -p 8000:8000 veh-dmg-backend
#
# The container serves the API only. A reachable Ollama instance running the VLM
# (configs/global_config.yaml -> vlm.model_id) is required separately.

FROM python:3.10-slim

WORKDIR /app

# System libraries required by OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install ML/pipeline deps and API deps.
COPY requirements-base.txt backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements-base.txt -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
