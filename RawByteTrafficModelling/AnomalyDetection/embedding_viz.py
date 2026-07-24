import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import umap

DATA_DIR = Path("RawByteTrafficModelling/AnomalyDetection/Outputs/PacketEmbeddings")
SEED = 42

# ---- load ----
X_parts, labels = [], []
for f in sorted(DATA_DIR.glob("*.npy")):
    arr = np.load(f)
    X_parts.append(arr)
    labels += [f.stem] * len(arr)

X = np.vstack(X_parts).astype(np.float32)
labels = np.array(labels)
cats = np.unique(labels)
print(X.shape, len(cats), "categories")

rng = np.random.default_rng(SEED)
N_MAX = 50_000
per_cat = max(1, N_MAX // len(cats))
idx = np.concatenate([rng.choice(np.where(labels == c)[0],
                                 min(per_cat, (labels == c).sum()), replace=False)
                      for c in cats])
X, labels = X[idx], labels[idx]

# L2-normalize (usually right for embeddings); comment out if not wanted
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
X = StandardScaler().fit_transform(X)

# PCA-50 pre-reduction speeds up t-SNE/UMAP a lot
# X50 = PCA(n_components=min(50, X.shape[1]), random_state=SEED).fit_transform(X)

# ---- reducers ----
methods = {
    # "PCA": lambda: PCA(n_components=2, random_state=SEED).fit_transform(X),
    # "t-SNE (perp=30)": lambda: TSNE(n_components=2, perplexity=30, init="pca",
    #                                 random_state=SEED).fit_transform(X50),
    # "UMAP (n=15, d=0.1)": lambda: umap.UMAP(n_neighbors=15, min_dist=0.1,
    #                                         metric="cosine", random_state=SEED).fit_transform(X50),
    "UMAP (n=50, d=0.1)": lambda: umap.UMAP(n_neighbors=50, min_dist=0.1,
                                            metric="cosine").fit_transform(X),
    "UMAP (n=100, d=0.01)": lambda: umap.UMAP(n_neighbors=100, min_dist=0.01,
                                            metric="cosine").fit_transform(X),
    "UMAP (n=300, d=0.0001)": lambda: umap.UMAP(n_neighbors=300, min_dist=0.0001,
                                            metric="cosine").fit_transform(X),
    "UMAP (n=1000, d=0.001)": lambda: umap.UMAP(n_neighbors=1000, min_dist=0.0001,
                                            metric="cosine").fit_transform(X),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
colors = plt.cm.tab20(np.linspace(0, 1, len(cats)))

for ax, (name, fn) in zip(axes.ravel(), methods.items()):
    Z = fn()
    for c, col in zip(cats, colors):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=6, alpha=0.6, color=col, label=c)
    ax.set_title(name)
    ax.set_xticks([]); ax.set_yticks([])
    print(f"{name}, done!")

axes[0, 0].legend(markerscale=3, fontsize=8, loc="best")
plt.tight_layout()
plt.savefig("embedding_projections.png", dpi=600)
plt.show()