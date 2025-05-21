from sklearn.cluster import KMeans
from .base import BaseClusterer

class KMeansClusterer(BaseClusterer):
    def cluster(self, texts, true_labels=None, n_clusters=2):
        X, coords = self._prepare_data(texts)
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        predicted = model.fit_predict(X)
        
        df = self._create_result_df(texts, coords, predicted, true_labels)
        accuracy = self._calculate_accuracy(true_labels, predicted)
            
        return df, accuracy 