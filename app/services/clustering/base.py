from abc import ABC, abstractmethod
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

class BaseClusterer(ABC):
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000)
        
    def _prepare_data(self, texts):
        X = self.vectorizer.fit_transform(texts)
        coords = TSNE(n_components=2, random_state=42, perplexity=50).fit_transform(X.toarray())
        return X, coords
        
    def _create_result_df(self, texts, coords, predicted, true_labels=None):
        df = pd.DataFrame({
            "text": texts,
            "x": coords[:, 0],
            "y": coords[:, 1],
            "cluster": predicted
        })
        
        if true_labels is not None:
            df["true_label"] = true_labels
            
        return df
        
    def _calculate_accuracy(self, true_labels, predicted):
        if true_labels is not None:
            return max(
                accuracy_score(true_labels, predicted),
                accuracy_score(true_labels, 1 - predicted)
            )
        return None
        
    @abstractmethod
    def cluster(self, texts, true_labels=None, n_clusters=2):
        pass 