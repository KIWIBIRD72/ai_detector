import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Загрузка модели
model_path = os.path.abspath("./app/services/detector/model")
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