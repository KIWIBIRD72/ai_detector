from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import pandas as pd
from datasets import load_dataset
from functools import lru_cache


dataset = load_dataset("HumanLLMs/Human-Like-DPO-Dataset")
train_data = dataset["train"].shuffle(seed=42)

split_ratio = 1
split_index = int(len(train_data) * split_ratio)
train_subset = train_data.select(range(split_index)) # split_ratio% от данных

def cluster_texts(texts, true_labels=None, n_clusters=2):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    predicted = kmeans.fit_predict(X)

    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    coords = tsne.fit_transform(X.toarray())

    df = pd.DataFrame({
        "text": texts,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "cluster": predicted
    })

    if true_labels is not None:
        df["true_label"] = true_labels
        accuracy = max(
            accuracy_score(true_labels, predicted),
            accuracy_score(true_labels, 1 - predicted)
        )
        return df, accuracy

    return df, None

# Кэшируем результат кластеризации (до перезапуска приложения)
@lru_cache(maxsize=1)
def get_cluster_data_cached():
    """
    0 - Human
    1 - AI
    """
    texts, labels = [], []
    for example in train_subset:
        texts.append(example["chosen"])
        labels.append(0)
        texts.append(example["rejected"])
        labels.append(1)

    df, acc = cluster_texts(texts, labels)
    return df.to_dict(orient="records"), acc


