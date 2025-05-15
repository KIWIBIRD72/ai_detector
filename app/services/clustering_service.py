import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import pandas as pd
from datasets import load_dataset
import numpy as np
import skfuzzy as fuzz

# Путь к папке с кэшами
CACHE_DIR = "cluster_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

# Загружаем датасет
dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")
train_data = dataset["train"].shuffle(seed=42)

# Используем 100% данных (можно изменить)
split_ratio = 1
split_index = int(len(train_data) * split_ratio)
train_subset = train_data.select(range(split_index))


def cluster_texts(texts, true_labels=None, n_clusters=2, method="kmeans"):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)
    coords = TSNE(n_components=2, random_state=42, perplexity=50).fit_transform(X.toarray())

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        predicted = model.fit_predict(X)
    elif method == "fuzzy":
        X_dense = X.toarray().T
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X_dense, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None, seed=42
        )
        predicted = np.argmax(u, axis=0)
    else:
        raise ValueError("Unsupported clustering method. Use 'kmeans' or 'fuzzy'.")

    df = pd.DataFrame({
        "text": texts,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": predicted
    })

    accuracy = None
    if true_labels is not None:
        df["true_label"] = true_labels
        accuracy = max(
            accuracy_score(true_labels, predicted),
            accuracy_score(true_labels, 1 - predicted)
        )

    return df, accuracy


def get_cluster_data_cached(method="kmeans"):
    cache_file = os.path.join(CACHE_DIR, f"clusters_{method}.json")

    # Если файл существует — читаем и возвращаем
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data["records"]
        accuracy = data.get("accuracy")
        return records, accuracy

    # Иначе — считаем и сохраняем
    texts, labels = [], []
    for example in train_subset:
        texts.append(example["chosen"])
        labels.append(0)
        texts.append(example["rejected"])
        labels.append(1)

    df, acc = cluster_texts(texts, labels, method=method)

    # Сохраняем в файл
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({
            "records": df.to_dict(orient="records"),
            "accuracy": acc
        }, f, ensure_ascii=False, indent=2)

    return df.to_dict(orient="records"), acc
