import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .train_service import train_model

# Загрузка модели
model_path = os.path.abspath("./app/services/detector/model")

# Проверка наличия модели и её создание при необходимости
if not os.path.exists(model_path) or not os.listdir(model_path):
    print("Model not found. Training new model...")
    train_model()
    print("Model training completed.")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# Функция предсказания
def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {"human": float(probabilities[0][0]), "ai": float(probabilities[0][1])}