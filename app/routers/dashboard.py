import os
from fastapi import APIRouter
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles

router = APIRouter(prefix="/dashboard", tags=["dashboard"])

DASHBOARD_STATIC_PATH = '/Users/evgenijtrubnikov/Developer/PyCharm/naturale_or_artificial_text/frontend/dist'
router.mount("/dashboard", StaticFiles(directory=DASHBOARD_STATIC_PATH, html=True), name="dashboard")

@router.get("/dashboard/{full_path:path}")
async def serve_spa(full_path: str):
    file_path = os.path.join(DASHBOARD_STATIC_PATH, "index.html")
    return FileResponse(file_path)