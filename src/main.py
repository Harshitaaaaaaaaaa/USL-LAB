from src.preprocessing import load_data, get_features, scale_data
from src.model import run_all_models
from src.persona import create_persona
from src.visualize import plot_all

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ==============================
# 1. LOAD AND PREPROCESS DATA
# ==============================
df = load_data("data/credit_card.csv")

X = get_features(df)
X_scaled = scale_data(X)


# ==============================
# 2. DETERMINE OPTIMAL K
# ==============================
best_k = 2
best_score = -1
k_scores = {}

print("\nFinding optimal number of clusters...\n")

for k in range(2, 10):
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    k_scores[k] = score
    print(f"k = {k}, silhouette score = {score:.3f}")

    if score > best_score:
        best_score = score
        best_k = k

print(f"\nBest k based on silhouette score: {best_k} (score = {best_score:.3f})")


# ==============================
# 3. FINAL K SELECTION (JUSTIFICATION)
# ==============================
final_k = 3

print("\nFinal Decision:")
print(f"Using k = {final_k} for clustering")

print("\nJustification:")
print(f"- Silhouette score suggests optimal k = {best_k}")
print("- However, k = 3 provides better interpretability")
print("- It segments customers into high, medium, and low value groups")
print("- This improves business decision-making and usability")

# Override k
best_k = final_k


# ==============================
# 4. APPLY CLUSTERING MODELS
# ==============================
results = run_all_models(X_scaled, k=best_k)

df['kmeans'] = results['kmeans']
df['agg'] = results['agg']
df['dbscan'] = results['dbscan']

# Final ensemble cluster
df['cluster'] = results['ensemble']


# ==============================
# 5. GENERATE CUSTOMER PERSONAS
# ==============================
features = X.columns.tolist()
df, summary = create_persona(df, features)

print("\nCluster Summary:\n", summary)


# ==============================
# 6. MODEL EVALUATION
# ==============================
print("\nSilhouette Scores:\n")

scores = {}

scores['KMeans'] = silhouette_score(X_scaled, df['kmeans'])
scores['Agglomerative'] = silhouette_score(X_scaled, df['agg'])

# DBSCAN evaluation with safety check
try:
    if len(set(df['dbscan'])) > 1:
        scores['DBSCAN'] = silhouette_score(X_scaled, df['dbscan'])
    else:
        scores['DBSCAN'] = "Not valid (single cluster)"
except:
    scores['DBSCAN'] = "Error"

# Ensemble evaluation
scores['Ensemble'] = silhouette_score(X_scaled, df['cluster'])

# Print results
for model, score in scores.items():
    print(f"{model}: {score}")

# Identify best model
valid_scores = {k: v for k, v in scores.items() if isinstance(v, float)}
best_model = max(valid_scores, key=valid_scores.get)

print(f"\nBest performing model: {best_model} ({valid_scores[best_model]:.3f})")


# ==============================
# 7. VISUALIZATION
# ==============================
plot_all(df, X_scaled, features)


# ==============================
# 8. SAVE OUTPUT
# ==============================
df.to_csv("results/outputs/final_output.csv", index=False)

print("\nProject execution completed successfully.")