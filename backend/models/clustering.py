import os
import json
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import entropy



EMBEDDINGS_PATH     = "./embeddings/embeddings.npy"
TEXTS_PATH          = "./embeddings/texts.json"
MODELS_DIR          = "./models"
GMM_MODEL_PATH      = "./models/gmm_model.joblib"
ASSIGNMENTS_PATH    = "./models/gmm_assignments.npy"
BIC_PLOT_PATH       = "./models/bic_curve.png"
PCA_MODEL_PATH      = "./models/pca_model.joblib"

K_RANGE             = range(5, 41, 5)   
PCA_COMPONENTS      = 50
RANDOM_STATE        = 42



def load_data():
    print(" Loading embeddings and texts...")

    embeddings = np.load(EMBEDDINGS_PATH)
    print(f" Embeddings loaded: {embeddings.shape}")

    with open(TEXTS_PATH, 'r', encoding='utf-8') as f:
        texts_data = json.load(f)

    texts  = [d['text']     for d in texts_data]
    labels = [d['label']    for d in texts_data]
    print(f" Texts loaded: {len(texts)} documents")

    return embeddings, texts, labels


#  PCA DIMENSIONALITY REDUCTION

def reduce_dimensions(embeddings):
    print(f"\n Reducing dimensions: {embeddings.shape[1]}D → {PCA_COMPONENTS}D with PCA...")

    pca = PCA(n_components=PCA_COMPONENTS, random_state=RANDOM_STATE)
    reduced = pca.fit_transform(embeddings)

    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"PCA complete. Variance retained: {variance_explained:.1f}%")
    print(f"   Reduced shape: {reduced.shape}")

    # Save PCA model for use in API later
    joblib.dump(pca, PCA_MODEL_PATH)
    print(f" PCA model saved to {PCA_MODEL_PATH}")

    return reduced, pca


def find_best_k(reduced):
    print(f"\n Finding optimal cluster count using BIC...")
    print(f"   Testing k values: {list(K_RANGE)}")
    print(f"    This will take several minutes...")

    bic_scores = []
    aic_scores = []

    for k in K_RANGE:
        print(f"   Fitting GMM with k={k}...", end=" ")
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=RANDOM_STATE,
            max_iter=100,
            n_init=1
        )
        gmm.fit(reduced)
        bic = gmm.bic(reduced)
        aic = gmm.aic(reduced)
        bic_scores.append(bic)
        aic_scores.append(aic)
        print(f"BIC={bic:.0f}")

    # Plot BIC curve
    plt.figure(figsize=(10, 5))
    plt.plot(list(K_RANGE), bic_scores, 'bo-', label='BIC', linewidth=2)
    plt.plot(list(K_RANGE), aic_scores, 'rs-', label='AIC', linewidth=2)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Score (lower is better)')
    plt.title('GMM Model Selection: BIC & AIC vs Number of Clusters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(BIC_PLOT_PATH)
    plt.close()
    print(f"\n BIC curve saved to {BIC_PLOT_PATH}")

    # Pick k at the elbow — where BIC improvement drops below 1% of total range
    bic_array   = np.array(bic_scores)
    bic_range   = bic_array.max() - bic_array.min()
    improvements = np.diff(bic_array) * -1  # positive = improvement
    threshold   = bic_range * 0.01          # 1% of total BIC range

    best_idx = 0
    for i, improvement in enumerate(improvements):
        if improvement > threshold:
            best_idx = i + 1  # keep updating as long as improvement is significant

    best_k = list(K_RANGE)[best_idx]
    print(f"\n Best k selected: {best_k}")
    print(f"   Reason: marginal BIC improvement drops below threshold after k={best_k}")

    return best_k, bic_scores



def fit_final_gmm(reduced, best_k):
    print(f"\n Fitting final GMM with k={best_k}...")

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type='full',
        random_state=RANDOM_STATE,
        max_iter=200,
        n_init=3,       # fit 3 times, keep best — more stable result
        verbose=1
    )
    gmm.fit(reduced)

    # Soft assignments — shape: (n_docs, best_k)
    # Each row is a probability distribution over clusters
    soft_assignments = gmm.predict_proba(reduced)
    hard_assignments = gmm.predict(reduced)

    print(f"\n GMM fitted successfully")
    print(f"   Soft assignments shape: {soft_assignments.shape}")
    print(f"   Example (first doc): {soft_assignments[0].round(3)}")

    # Save model and assignments
    joblib.dump(gmm, GMM_MODEL_PATH)
    np.save(ASSIGNMENTS_PATH, soft_assignments)
    print(f" GMM model saved to {GMM_MODEL_PATH}")
    print(f" Soft assignments saved to {ASSIGNMENTS_PATH}")

    return gmm, soft_assignments, hard_assignments



def analyze_clusters(texts, soft_assignments, hard_assignments, best_k):
    print(f"\n Analyzing clusters...")

    # --- Top TF-IDF terms per cluster ---
    print("\n Top TF-IDF terms per cluster:")
    print("-" * 50)

    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    cluster_terms = {}
    for cluster_id in range(best_k):
        # Weight TF-IDF by GMM soft probability for this cluster
        weights         = soft_assignments[:, cluster_id]
        weighted_tfidf  = np.array(tfidf_matrix.T.dot(weights)).flatten()
        top_indices     = weighted_tfidf.argsort()[-10:][::-1]
        top_terms       = [feature_names[i] for i in top_indices]
        cluster_terms[cluster_id] = top_terms

        # Count dominant docs in this cluster
        dominant_count = (hard_assignments == cluster_id).sum()
        print(f"\n  Cluster {cluster_id} ({dominant_count} docs dominant):")
        print(f"  Terms: {', '.join(top_terms)}")

    # --- Boundary Documents (highest entropy = most uncertain) ---
    print(f"\n\n Boundary Documents (most uncertain — highest entropy):")
    print("-" * 50)

    doc_entropy = entropy(soft_assignments.T)  # entropy per document
    top_boundary_idx = np.argsort(doc_entropy)[-5:][::-1]

    for idx in top_boundary_idx:
        probs       = soft_assignments[idx]
        top2_clusters = np.argsort(probs)[-2:][::-1]
        print(f"\n  Doc {idx} (entropy={doc_entropy[idx]:.3f}):")
        print(f"  Top clusters: {top2_clusters[0]} ({probs[top2_clusters[0]]:.2f}) "
              f"& {top2_clusters[1]} ({probs[top2_clusters[1]]:.2f})")
        print(f"  Text snippet: {texts[idx][:150]}...")

    # --- Cluster Size Distribution ---
    print(f"\n\n Cluster Size Distribution:")
    print("-" * 50)
    for cluster_id in range(best_k):
        count = (hard_assignments == cluster_id).sum()
        bar   = "" * (count // 100)
        print(f"  Cluster {cluster_id:2d}: {count:5d} docs  {bar}")

    # --- Mean Intra-cluster Cosine Similarity ---
    print(f"\n\n Semantic Coherence (mean intra-cluster cosine similarity):")
    print("-" * 50)
    embeddings = np.load(EMBEDDINGS_PATH)
    for cluster_id in range(best_k):
        mask        = hard_assignments == cluster_id
        cluster_emb = embeddings[mask]
        if len(cluster_emb) > 1:
            # Sample max 200 docs for speed
            sample = cluster_emb[:200]
            sim_matrix = sample @ sample.T
            # Mean of upper triangle (excluding diagonal)
            upper = sim_matrix[np.triu_indices(len(sample), k=1)]
            mean_sim = upper.mean()
            print(f"  Cluster {cluster_id:2d}: {mean_sim:.4f}")

    return cluster_terms, doc_entropy



def main():
    print("=" * 55)
    print("  PART 2 — Fuzzy Clustering with GMM")
    print("=" * 55)

    os.makedirs(MODELS_DIR, exist_ok=True)

    embeddings, texts, labels       = load_data()
    reduced, pca                    = reduce_dimensions(embeddings)
    best_k, bic_scores              = find_best_k(reduced)
    gmm, soft_assignments, hard_assignments = fit_final_gmm(reduced, best_k)
    cluster_terms, doc_entropy      = analyze_clusters(
                                        texts, soft_assignments,
                                        hard_assignments, best_k
                                      )




if __name__ == "__main__":
    main()