from fastapi import APIRouter
from fastapi import UploadFile, File
from app.services.detector.predict import predict

router = APIRouter(prefix="/detector", tags=["detector"])

@router.post("/check")
async def check_text(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    result = predict(text)
    return {"file_name": file.filename, "ai_probability": result["ai"], "human_probability": result["human"]}
