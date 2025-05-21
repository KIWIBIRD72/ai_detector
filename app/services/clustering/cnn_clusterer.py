import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from .base import BaseClusterer
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CNNClusterer(BaseClusterer):
    def __init__(self):
        super().__init__()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        logger.info("Loading BERT model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def _get_embeddings(self, texts):
        embeddings = []
        batch_size = 128
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(texts)} texts in {total_batches} batches")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                  max_length=512, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
                
        return np.vstack(embeddings)
        
    def cluster(self, texts, true_labels=None, n_clusters=2):
        logger.info("Starting clustering process...")
        
        # Получаем эмбеддинги из BERT
        logger.info("Generating BERT embeddings...")
        X = self._get_embeddings(texts)
        
        # Применяем t-SNE для визуализации
        logger.info("Applying t-SNE for visualization...")
        coords = self._prepare_data(texts)[1]
        
        # Кластеризуем эмбеддинги
        logger.info(f"Performing KMeans clustering with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        predicted = kmeans.fit_predict(X)
        
        logger.info("Creating result dataframe...")
        df = self._create_result_df(texts, coords, predicted, true_labels)
        accuracy = self._calculate_accuracy(true_labels, predicted)
        
        logger.info("Clustering completed successfully")
        return df, accuracy 