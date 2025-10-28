"""Clustering module for unsupervised grouping of embeddings."""

import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClusteringEngine:
    """Perform clustering on embeddings."""
    
    def __init__(self, method: str = 'kmeans', n_clusters: int = 5):
        self.method = method
        self.n_clusters = n_clusters
        self.model = None
        self.labels_ = None
        self.embeddings_2d = None
    
    def fit(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit clustering model on embeddings."""
        logger.info(f"Fitting {self.method} clustering with {self.n_clusters} clusters")
        
        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.method == 'dbscan':
            self.model = DBSCAN(eps=0.5, min_samples=5)
        elif self.method == 'hierarchical':
            self.model = AgglomerativeClustering(n_clusters=self.n_clusters)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        self.labels_ = self.model.fit_predict(embeddings)
        logger.info(f"Clustering complete. Found {len(np.unique(self.labels_))} clusters")
        
        return self.labels_
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new embeddings."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if hasattr(self.model, 'predict'):
            return self.model.predict(embeddings)
        else:
            # For models without predict method, use fit_predict
            return self.model.fit_predict(embeddings)
    
    def reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce dimensionality for visualization using PCA."""
        pca = PCA(n_components=n_components)
        self.embeddings_2d = pca.fit_transform(embeddings)
        logger.info(f"Reduced dimensions to {n_components}D. Explained variance: {pca.explained_variance_ratio_.sum():.2%}")
        return self.embeddings_2d
    
    def visualize_clusters(self, embeddings: np.ndarray, labels: np.ndarray, 
                          output_path: str = 'data/visualizations/clusters.png'):
        """Visualize clusters in 2D space."""
        if embeddings.shape[1] > 2:
            embeddings_2d = self.reduce_dimensions(embeddings, n_components=2)
        else:
            embeddings_2d = embeddings
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=labels, cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label='Cluster')
        plt.title(f'{self.method.upper()} Clustering Results')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True, alpha=0.3)
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Cluster visualization saved to {output_path}")
    
    def get_cluster_stats(self, labels: np.ndarray) -> Dict:
        """Get statistics about clusters."""
        unique_labels = np.unique(labels)
        stats = {
            'n_clusters': len(unique_labels),
            'cluster_sizes': {},
            'largest_cluster': None,
            'smallest_cluster': None
        }
        
        for label in unique_labels:
            size = np.sum(labels == label)
            stats['cluster_sizes'][int(label)] = int(size)
        
        sizes = list(stats['cluster_sizes'].values())
        stats['largest_cluster'] = max(sizes) if sizes else 0
        stats['smallest_cluster'] = min(sizes) if sizes else 0
        
        logger.info(f"Cluster statistics: {stats}")
        return stats
    
    def find_cluster_centers(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[int, np.ndarray]:
        """Find the center (centroid) of each cluster."""
        centers = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            cluster_embeddings = embeddings[labels == label]
            center = np.mean(cluster_embeddings, axis=0)
            centers[int(label)] = center
        
        return centers
    
    def get_items_in_cluster(self, cluster_id: int, labels: np.ndarray) -> List[int]:
        """Get indices of all items in a specific cluster."""
        indices = np.where(labels == cluster_id)[0]
        return indices.tolist()

if __name__ == "__main__":
    # Example usage
    engine = ClusteringEngine(method='kmeans', n_clusters=5)
    print("Clustering module loaded successfully")
