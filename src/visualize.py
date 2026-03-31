import matplotlib
matplotlib.use('TkAgg')  # Ensure plots open in separate windows

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def plot_all(df, X_scaled, features):

    # Set plotting style
    sns.set(style="whitegrid")

    # ==============================
    # 1. BASIC DASHBOARD
    # ==============================
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Scatter plot: Balance vs Payments
    scatter = axs[0, 0].scatter(
        df['BALANCE'],
        df['PAYMENTS'],
        c=df['cluster'],
        cmap='viridis'
    )
    axs[0, 0].set_title("Customer Segmentation (Balance vs Payments)")
    axs[0, 0].set_xlabel("Balance (Credit Balance)")
    axs[0, 0].set_ylabel("Payments (Amount Paid)")
    axs[0, 0].legend(*scatter.legend_elements(), title="Cluster Labels")

    # Bar chart: Persona distribution
    df['persona'].value_counts().plot(
        kind='bar',
        ax=axs[0, 1],
        color='skyblue'
    )
    axs[0, 1].set_title("Persona Distribution")
    axs[0, 1].set_xlabel("Persona Category")
    axs[0, 1].set_ylabel("Number of Customers")

    # Histogram: Balance distribution
    df['BALANCE'].hist(ax=axs[1, 0], color='orange')
    axs[1, 0].set_title("Balance Distribution")
    axs[1, 0].set_xlabel("Balance")
    axs[1, 0].set_ylabel("Frequency")

    # Bar chart: Average balance and payments per persona
    df.groupby('persona')[['BALANCE', 'PAYMENTS']].mean().plot(
        kind='bar',
        ax=axs[1, 1]
    )
    axs[1, 1].set_title("Average Balance and Payments by Persona")
    axs[1, 1].set_xlabel("Persona Category")
    axs[1, 1].set_ylabel("Average Value")

    plt.tight_layout()
    plt.savefig("results/plots/basic_plots.png")
    plt.show(block=True)

    # ==============================
    # 2. FEATURE CORRELATION HEATMAP
    # ==============================
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        df[features].corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )
    plt.title("Feature Correlation Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Features")

    plt.savefig("results/plots/heatmap.png")
    plt.show(block=True)

    # ==============================
    # 3. PCA TRANSFORMATION
    # ==============================
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_scaled)

    # ==============================
    # 4. CLUSTER VISUALIZATION (PCA)
    # ==============================
    models = ['kmeans', 'agg', 'dbscan', 'cluster']

    plt.figure(figsize=(16, 10))

    for i, model in enumerate(models):
        plt.subplot(2, 2, i + 1)

        scatter = plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=df[model],
            cmap='viridis'
        )

        try:
            if len(set(df[model])) > 1:
                score = silhouette_score(X_scaled, df[model])
                title = f"{model.upper()} (Silhouette = {score:.2f})"
            else:
                title = f"{model.upper()} (Invalid Clustering)"
        except:
            title = f"{model.upper()} (Error in Calculation)"

        plt.title(title)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(*scatter.legend_elements(), title="Cluster Labels")

    plt.tight_layout()
    plt.savefig("results/plots/model_clusters_pca.png")
    plt.show(block=True)

    # ==============================
    # 5. SILHOUETTE SCORE COMPARISON
    # ==============================
    scores = {}

    for model in models:
        try:
            if len(set(df[model])) > 1:
                scores[model] = silhouette_score(X_scaled, df[model])
        except:
            continue

    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=list(scores.keys()),
        y=list(scores.values())
    )
    plt.title("Silhouette Score Comparison Across Models")
    plt.xlabel("Clustering Models")
    plt.ylabel("Silhouette Score")

    # Display score values on bars
    for i, value in enumerate(scores.values()):
        plt.text(i, value + 0.01, f"{value:.2f}", ha='center')

    plt.savefig("results/plots/silhouette_comparison.png")
    plt.show(block=True)