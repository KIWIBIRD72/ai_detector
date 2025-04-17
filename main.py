import torch
from fastapi import FastAPI, UploadFile, File
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Функция предсказания
def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {"human": float(probabilities[0][0]), "ai": float(probabilities[0][1])}
app = FastAPI()

# Загрузка модели
model_path = "./app/services/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

@app.get("/")
def read_root():
    return {"message": "Hello from FastApi service"}

@app.post("/detector/check")
async def check_text(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    result = predict(text)
    return {"file_name": file.filename, "ai_probability": result["ai"], "human_probability": result["human"]}