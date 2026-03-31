import numpy as np
from collections import Counter
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


# ==============================
# 1. CLUSTERING MODELS
# ==============================

def apply_kmeans(X, k=3):
    """
    Apply KMeans clustering.
    """
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)

    print("\nKMeans cluster labels generated.")
    return labels


def apply_agglomerative(X, k=3):
    """
    Apply Agglomerative (hierarchical) clustering.
    """
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X)

    print("\nAgglomerative clustering labels generated.")
    return labels


def apply_dbscan(X):
    """
    Apply DBSCAN clustering.
    """
    model = DBSCAN(eps=1.2, min_samples=5)
    labels = model.fit_predict(X)

    # Replace noise points (-1) with a new cluster label
    if -1 in labels:
        labels[labels == -1] = max(labels) + 1

    print("\nDBSCAN clustering labels generated.")
    return labels


# ==============================
# 2. ENSEMBLE METHOD
# ==============================

def majority_voting(labels_list):
    """
    Combine clustering results using majority voting.
    """
    final_labels = []

    for i in range(len(labels_list[0])):
        votes = [labels[i] for labels in labels_list]
        most_common_label = Counter(votes).most_common(1)[0][0]
        final_labels.append(most_common_label)

    final_labels = np.array(final_labels)

    print("\nEnsemble clustering completed using majority voting.")
    return final_labels


# ==============================
# 3. MAIN PIPELINE FUNCTION
# ==============================

def run_all_models(X, k=3):
    """
    Run all clustering models and return their labels.
    """
    print("\nRunning clustering models...\n")

    kmeans_labels = apply_kmeans(X, k)
    agg_labels = apply_agglomerative(X, k)
    dbscan_labels = apply_dbscan(X)

    # Ensemble using selected models
    labels_list = [
        kmeans_labels,
        agg_labels,
        dbscan_labels
    ]

    ensemble_labels = majority_voting(labels_list)

    # Summary of clusters
    print("\nClustering Summary:")
    print("KMeans clusters:", np.unique(kmeans_labels))
    print("Agglomerative clusters:", np.unique(agg_labels))
    print("DBSCAN clusters:", np.unique(dbscan_labels))
    print("Ensemble clusters:", np.unique(ensemble_labels))

    return {
        "kmeans": kmeans_labels,
        "agg": agg_labels,
        "dbscan": dbscan_labels,
        "ensemble": ensemble_labels
    }