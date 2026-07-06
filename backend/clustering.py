import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN

_embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def cluster_posts(posts: list, eps: float = 0.35, min_samples: int = 2):
    """
    Groups posts into narrative clusters using sentence embeddings + DBSCAN.
    Posts in the same cluster are treated as spreading the same narrative,
    even if reworded differently.
    """
    texts = [p["text"] for p in posts if p.get("text")]
    valid_posts = [p for p in posts if p.get("text")]

    if len(texts) < 2:
        return {"clusters": [], "note": "Not enough posts to cluster"}

    embeddings = _embed_model.encode(texts)

    # DBSCAN uses distance, so we convert cosine similarity to cosine distance
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings)
    labels = clustering.labels_

    clusters = {}
    for post, label in zip(valid_posts, labels):
        if label == -1:
            continue  # noise / doesn't belong to any narrative cluster
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