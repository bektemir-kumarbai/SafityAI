from fastapi import APIRouter
from .detect import router as action_router

router = APIRouter(prefix="/api/v1", tags=["actions"])
router.include_router(action_router)
