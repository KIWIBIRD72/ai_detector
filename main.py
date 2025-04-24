import torch
from fastapi import FastAPI, UploadFile, File
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.services.clustering_service import cluster_texts, get_cluster_data_cached
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware


# Функция предсказания
def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return {"human": float(probabilities[0][0]), "ai": float(probabilities[0][1])}
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.get("/clusters")
def get_clusters(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100)
):
    data, accuracy = get_cluster_data_cached()

    total = len(data)
    start = (page - 1) * page_size
    end = start + page_size

    paginated_data = data[start:end]

    return {
        "accuracy": accuracy,
        "total": total,
        "page": page,
        "page_size": page_size,
        "data": paginated_data
    }

@app.get("/clusters/stats")
def get_stats():
    _, accuracy = get_cluster_data_cached()
    return {"accuracy": accuracy}