import numpy as np
from sklearn.cluster import DBSCAN

_embed_model = None

def _get_embed_model():
    from sentence_transformers import SentenceTransformer
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model

def cluster_posts(posts: list, eps: float = 0.35, min_samples: int = 2):
    texts = [p["text"] for p in posts if p.get("text")]
    valid_posts = [p for p in posts if p.get("text")]

    if len(texts) < 2:
        return {"clusters": [], "note": "Not enough posts to cluster"}

    embeddings = _get_embed_model().encode(texts)

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_

    clusters = {}
    for post, label in zip(valid_posts, labels):
        if label == -1:
            continue
        clusters.setdefault(int(label), []).append(post)

    result = []
    for cluster_id, cluster_posts in clusters.items():
        sorted_posts = sorted(cluster_posts, key=lambda x: x["timestamp"])
        result.append({
            "cluster_id": cluster_id,
            "size": len(sorted_posts),
            "earliest_post": sorted_posts[0],
            "posts": sorted_posts
        })

    result.sort(key=lambda x: x["size"], reverse=True)
    return {"clusters": result}