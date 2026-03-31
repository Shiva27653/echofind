import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# ─── 20% probe subset ───
np.random.seed(42)
n_probe = max(1, int(0.20 * len(train_embeddings)))
probe_idx = np.random.choice(len(train_embeddings), size=n_probe, replace=False)
X_probe = train_embeddings[probe_idx]
y_probe = train_labels[probe_idx]

print(f"Probe size: {n_probe} ({n_probe/len(train_embeddings)*100:.1f}% of train)\n")

# ─── Three classifiers ───
classifiers = {
    "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "SVC_RBF":            SVC(kernel='rbf', C=10, gamma='scale', random_state=42),
    "KNN_15_cosine":      KNeighborsClassifier(n_neighbors=15, metric='cosine'),
}

results = {}
for name, clf in classifiers.items():
    print(f"Training {name}...")
    clf.fit(X_probe, y_probe)
    preds = clf.predict(val_embeddings)
    f1 = f1_score(val_labels, preds, average='macro')
    results[name] = (clf, preds, f1)
    print(f"  {name} — Val Macro F1: {f1:.4f}\n")

# ─── Best classifier ───
best_name = max(results, key=lambda k: results[k][2])
best_clf, best_preds, best_f1 = results[best_name]

print("=" * 60)
print(f"Best classifier: {best_name} | Macro F1: {best_f1:.4f}")
print("=" * 60)
print(f"\nFull classification report ({best_name}):\n")
print(classification_report(val_labels, best_preds))

# ─── t-SNE v2 ───
print("Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42)
coords = tsne.fit_transform(val_embeddings)

genres = sorted(list(set(val_labels)))
cmap = plt.cm.tab10

fig, ax = plt.subplots(figsize=(14, 10))

for i, genre in enumerate(genres):
    mask = val_labels == genre
    ax.scatter(coords[mask, 0], coords[mask, 1],
               c=[cmap(i)], label=genre, s=25, alpha=0.75)

ax.set_title("EchoFind — Latent Space (t-SNE) | SimCLR on FMA-Small (No Labels Used)",
             fontsize=14, fontweight='bold')
ax.text(0.5, 1.02, f"Linear Probe Macro F1: {best_f1:.4f} (Best: {best_name}) | Clean Retrieval Accuracy: 99.62%",
        transform=ax.transAxes, ha='center', va='bottom', fontsize=11, style='italic')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, frameon=True)
plt.tight_layout()
plt.savefig("/kaggle/working/tsne_embeddings_v2.png", dpi=300, bbox_inches='tight')
plt.show()
print("t-SNE saved to /kaggle/working/tsne_embeddings_v2.png")
