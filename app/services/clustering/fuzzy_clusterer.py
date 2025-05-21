import numpy as np
import skfuzzy as fuzz
from .base import BaseClusterer

class FuzzyClusterer(BaseClusterer):
    def cluster(self, texts, true_labels=None, n_clusters=2):
        X, coords = self._prepare_data(texts)
        X_dense = X.toarray().T
        
        cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
            X_dense, c=n_clusters, m=2, error=0.005, maxiter=1000, init=None, seed=42
        )
        predicted = np.argmax(u, axis=0)
        
        df = self._create_result_df(texts, coords, predicted, true_labels)
        accuracy = self._calculate_accuracy(true_labels, predicted)
            
        return df, accuracy 