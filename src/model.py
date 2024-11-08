from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np

def train_model(X, n_clusters=5):
    # Train KMeans model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    
    return kmeans

def find_optimal_clusters(X, max_clusters=10):
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores.append(silhouette_avg)
    
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
    return optimal_clusters

def get_cluster_centers(model, scaler):
    # Get cluster centers and inverse transform to original scale
    centers = model.cluster_centers_
    centers_original = scaler.inverse_transform(centers[:, :4])  # Only transform numeric features
    
    return centers_original