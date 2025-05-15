import torch
from fastapi import FastAPI, UploadFile, File
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.services.clustering_service import get_cluster_data_cached
from fastapi import Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import os
from app.services.ner.nltk_ner_service import NltkNer


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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели
model_path = "./app/services/model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

DASHBOARD_STATIC_PATH = '/Users/evgenijtrubnikov/Developer/PyCharm/naturale_or_artificial_text/frontend/dist'
app.mount("/dashboard", StaticFiles(directory=DASHBOARD_STATIC_PATH, html=True), name="dashboard")

@app.get('/ner-entities')
async def get_entities(text: str):
    nltk_ner = NltkNer()
    nltk_named_entities = nltk_ner.get_named_entities(text)
    return nltk_named_entities

# Для поддержки SPA роутинга
@app.get("/dashboard/{full_path:path}")
async def serve_spa(full_path: str):
    file_path = os.path.join(DASHBOARD_STATIC_PATH, "index.html")
    return FileResponse(file_path)

@app.post("/detector/check")
async def check_text(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")
    result = predict(text)
    return {"file_name": file.filename, "ai_probability": result["ai"], "human_probability": result["human"]}


@app.get("/clusters")
def get_clusters(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=300),
    method: str = Query("kmeans", enum=["kmeans", "fuzzy"]),
    n_cluster: int = Query(2, ge=1)
):
    data, accuracy = get_cluster_data_cached(method, n_cluster)

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
def get_stats(method: str = Query("kmeans", enum=["kmeans", "fuzzy"])):
    _, accuracy = get_cluster_data_cached(method)
    return {"accuracy": accuracy}