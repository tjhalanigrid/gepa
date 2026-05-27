from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import assessment

app = FastAPI(
    title="Vehicle Damage AI API Service",
    description="Microservice handling damage classification, panel segmentations, and cost estimations.",
    version="1.0.0"
)

# Enable CORS for frontend dashboard queries
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include claim assessment router
app.include_router(assessment.router, prefix="/api")

@app.get("/")
def root():
    """
    Service health check route.
    """
    return {
        "status": "RUNNING",
        "service": "Vehicle Damage AI API Engine",
        "version": "1.0.0"
    }