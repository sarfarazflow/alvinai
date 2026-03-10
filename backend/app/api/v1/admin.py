from fastapi import APIRouter, Depends, HTTPException
from app.api.deps import get_current_user
from app.models.user import User

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats")
async def stats(user: User = Depends(get_current_user)):
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return {"status": "ok", "message": "Admin stats endpoint — coming soon"}
