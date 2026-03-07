from fastapi import FastAPI
from app.api.detect import router

app = FastAPI(
    title="MMAction API",
    description="API for video action detection",
    version="1.0.0",
)

app.include_router(router)