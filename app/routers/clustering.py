from fastapi import APIRouter, Query
from app.services.clustering.kmeans_service import get_kmeans_cluster_data
from app.services.clustering.fuzzy_service import get_fuzzy_cluster_data
from app.services.clustering.cnn_service import get_cnn_cluster_data

router = APIRouter(prefix="/clusters", tags=["clustering"])

@router.get("/kmeans")
def get_kmeans_clusters(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=300),
    n_cluster: int = Query(2, ge=1)
):
    data, accuracy = get_kmeans_cluster_data(n_cluster)
    
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

@router.get("/fuzzy")
def get_fuzzy_clusters(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=300),
    n_cluster: int = Query(2, ge=1)
):
    data, accuracy = get_fuzzy_cluster_data(n_cluster)
    
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

@router.get("/cnn")
def get_cnn_clusters(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=300),
    n_cluster: int = Query(2, ge=1)
):
    data, accuracy = get_cnn_cluster_data(n_cluster)
    
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

@router.get("/stats")
def get_stats(method: str = Query("kmeans", enum=["kmeans", "fuzzy", "cnn"])):
    if method == "kmeans":
        _, accuracy = get_kmeans_cluster_data()
    elif method == "fuzzy":
        _, accuracy = get_fuzzy_cluster_data()
    else:
        _, accuracy = get_cnn_cluster_data()
    return {"accuracy": accuracy}
