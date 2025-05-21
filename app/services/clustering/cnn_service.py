import os
import json
from datasets import load_dataset
from app.services.clustering.cnn_clusterer import CNNClusterer

CACHE_DIR = os.path.abspath("cluster_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")
train_data = dataset["train"].shuffle(seed=42)
split_ratio = 1
split_index = int(len(train_data) * split_ratio)
train_subset = train_data.select(range(split_index))

def get_cnn_cluster_data(n_cluster=2):
    cache_file = os.path.join(CACHE_DIR, f"clusters_cnn_{n_cluster}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["records"], data.get("accuracy")

    texts, labels = [], []
    for example in train_subset:
        texts.append(example["chosen"])
        labels.append(0)
        texts.append(example["rejected"])
        labels.append(1)

    clusterer = CNNClusterer()
    df, acc = clusterer.cluster(texts, labels, n_cluster)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({
            "records": df.to_dict(orient="records"),
            "accuracy": acc
        }, f, ensure_ascii=False, indent=2)

    return df.to_dict(orient="records"), acc 