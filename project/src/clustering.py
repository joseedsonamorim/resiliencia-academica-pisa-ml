import numpy as np
import pandas as pd
import logging
import gc
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)

class ClusteringAnalyzer:
    def __init__(self, X: pd.DataFrame, n_clusters: int = 3):
        self.X = X
        self.n_clusters = n_clusters
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = None
        self.clusters = None

    def apply_pca(self, n_components: int = 20) -> np.ndarray:
        """Apply PCA with memory optimization."""
        try:
            # Standardize features
            X_scaled = self.scaler.fit_transform(self.X)

            # Apply PCA
            n_components = min(n_components, self.X.shape[1])
            self.pca = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca.fit_transform(X_scaled)

            explained_var = np.sum(self.pca.explained_variance_ratio_)
            self.logger.info(f"PCA: {n_components} components, {100*explained_var:.2f}% variance explained")
            gc.collect()
            return X_pca

        except Exception as e:
            self.logger.error(f"Error applying PCA: {str(e)}")
            raise

    def apply_kmeans(self, X_pca: np.ndarray) -> np.ndarray:
        """Apply KMeans clustering."""
        try:
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            self.clusters = self.kmeans.fit_predict(X_pca)
            self.logger.info(f"KMeans clustering completed. Clusters: {np.bincount(self.clusters)}")
            gc.collect()
            return self.clusters

        except Exception as e:
            self.logger.error(f"Error applying KMeans: {str(e)}")
            raise

    def calculate_metrics(self, X_pca: np.ndarray) -> dict:
        """Calculate clustering quality metrics."""
        try:
            silhouette = silhouette_score(X_pca, self.clusters)
            davies_bouldin = davies_bouldin_score(X_pca, self.clusters)

            metrics = {
                'silhouette_score': float(silhouette),
                'davies_bouldin_score': float(davies_bouldin),
                'n_clusters': self.n_clusters,
                'cluster_distribution': np.bincount(self.clusters).tolist()
            }

            self.logger.info(f"Silhouette: {silhouette:.4f}, Davies-Bouldin: {davies_bouldin:.4f}")
            return metrics

        except Exception as e:
            self.logger.error(f"Error calculating clustering metrics: {str(e)}")
            raise

    def plot_clusters(self, X_pca: np.ndarray, output_path: str = "project/outputs/figures/clusters.png"):
        """Plot PCA clusters."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=self.clusters, cmap='viridis', s=30, alpha=0.6)
            ax.scatter(self.kmeans.cluster_centers_[:, 0], self.kmeans.cluster_centers_[:, 1],
                      c='red', marker='X', s=200, edgecolors='black', label='Centroids')
            ax.set_xlabel(f'PC1 ({100*self.pca.explained_variance_ratio_[0]:.1f}%)')
            ax.set_ylabel(f'PC2 ({100*self.pca.explained_variance_ratio_[1]:.1f}%)')
            ax.set_title('Student Clusters (PCA + KMeans)')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            ax.legend()

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            self.logger.info(f"Clusters plot saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Error plotting clusters: {str(e)}")

    def analyze_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze cluster characteristics."""
        try:
            df_clusters = df.copy()
            df_clusters['Cluster'] = self.clusters

            analysis = df_clusters.groupby('Cluster').agg({
                'Creative_Resilience': ['sum', 'mean'],
                'ESCS': 'mean',
                'ST004D01T': 'mean'
            }).round(4)

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing clusters: {str(e)}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ClusteringAnalyzer module loaded successfully")
