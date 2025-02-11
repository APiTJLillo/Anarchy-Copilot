"""
FastAPI application initialization and configuration.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .health import router as health_router
from .proxy import router as proxy_router

app = FastAPI(
    title="Anarchy Copilot API",
    description="API for managing bug bounty operations",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health_router)
app.include_router(proxy_router)

@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    # Initialize any required resources
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    # Clean up any resources
    from .proxy import proxy_server
    if proxy_server:
        await proxy_server.stop()
