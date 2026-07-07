from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
#from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect

from sample_data import SAMPLE_POSTS
from scraper import scrape_reddit, scrape_youtube_comments
from hate_model import hate_speech_score, classify_threat
#from clustering import cluster_posts
from report_generator import generate_investigation_report

app = FastAPI(title="OriginAI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# _embed_model = None

# def get_embed_model():
#     from sentence_transformers import SentenceTransformer
#     global _embed_model
#     if _embed_model is None:
#         _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
#     return _embed_model
# In-memory store of collected posts (sample + real scraped data merged)
COLLECTED_POSTS = list(SAMPLE_POSTS)


@app.get("/")
def root():
    return {"status": "OriginAI backend is live"}


@app.get("/collect-live-data")
def collect_live_data(query: str = Query(..., description="Topic/keyword to search")):
    """
    Pulls REAL live posts from Reddit (and YouTube if a video_id is added)
    matching the given query, and adds them to the working dataset.
    """
    reddit_posts = scrape_reddit(query)
    COLLECTED_POSTS.extend(reddit_posts)
    return {
        "source": "reddit",
        "query": query,
        "new_posts_collected": len(reddit_posts),
        "posts": reddit_posts
    }


@app.get("/collect-youtube")
def collect_youtube(video_id: str):
    comments = scrape_youtube_comments(video_id)
    if isinstance(comments, dict) and "error" in comments:
        return comments
    COLLECTED_POSTS.extend(comments)
    return {"source": "youtube", "new_posts_collected": len(comments), "posts": comments}


@app.get("/sample-posts")
def get_all_posts():
    """Returns everything collected so far — sample + real scraped data."""
    return {"posts": COLLECTED_POSTS, "total": len(COLLECTED_POSTS)}


@app.post("/analyze-text")
def analyze_text(text: str = Form(...)):
    hate_result = hate_speech_score(text)
    threat_level = classify_threat(hate_result["overall_score"])

    try:
        language = detect(text)
    except Exception:
        language = "unknown"

def keyword_overlap_score(text_a: str, text_b: str) -> float:
              
            
				stopwords = {"the", "a", "an", "is", "in", "on", "at", "to", "and", "of", "for", "has", "have"}
				words_a = set(w.lower() for w in text_a.split() if w.lower() not in stopwords and len(w) > 2)
				words_b = set(w.lower() for w in text_b.split() if w.lower() not in stopwords and len(w) > 2)
				if not words_a or not words_b:
					return 0.0
				overlap = words_a & words_b
				return round(len(overlap) / min(len(words_a), len(words_b)), 3)

matches = []
valid_posts = [p for p in COLLECTED_POSTS if p.get("text")]
for post in valid_posts:
    sim = keyword_overlap_score(text, post["text"])
    matches.append({
        "id": post["id"],
        "platform": post["platform"],
        "user": post["user"],
        "text": post["text"],
        "timestamp": post["timestamp"],
        "similarity": sim
    })

relevant_matches = [m for m in matches if m["similarity"] > 0.3]
relevant_matches.sort(key=lambda x: x["similarity"], reverse=True)

    likely_originator = None
    if relevant_matches:
        earliest = min(relevant_matches, key=lambda x: x["timestamp"])
        avg_sim = sum(m["similarity"] for m in relevant_matches) / len(relevant_matches)
        confidence = round(min(0.99, avg_sim + 0.15), 2)
        likely_originator = {
            "user": earliest["user"],
            "platform": earliest["platform"],
            "timestamp": earliest["timestamp"],
            "confidence": confidence
        }

    return {
        "input_text": text,
        "language_detected": language,
        "hate_speech": hate_result,
        "threat_level": threat_level,
        "matched_posts": relevant_matches,
        "likely_originator": likely_originator,
        "evidence_count": len(relevant_matches)
    }


import tempfile
import os

# _ocr_reader = None

# def get_ocr_reader():
#     global _ocr_reader
#     if _ocr_reader is None:
#         import easyocr
#         _ocr_reader = easyocr.Reader(["en", "hi"], gpu=False)
#     return _ocr_reader

# @app.post("/analyze-image")
# async def analyze_image(file: UploadFile = File(...)):
#     reader = get_ocr_reader()
#     contents = await file.read()

#     temp_path = os.path.join(tempfile.gettempdir(), "temp_upload.png")
#     with open(temp_path, "wb") as f:
#         f.write(contents)

#     result = reader.readtext(temp_path, detail=0)
#     extracted_text = " ".join(result)
#     os.remove(temp_path)

#     if not extracted_text.strip():
#         return {"error": "No readable text found in image"}

#     return analyze_text(text=extracted_text)


@app.get("/propagation-graph")
def propagation_graph():
    sorted_posts = sorted(
        [p for p in COLLECTED_POSTS if p.get("timestamp")],
        key=lambda x: x["timestamp"]
    )
    nodes = [
        {"id": p["id"], "label": f'{p["platform"]} - {p["user"]}', "timestamp": p["timestamp"]}
        for p in sorted_posts
    ]
    edges = [
        {"source": sorted_posts[i]["id"], "target": sorted_posts[i + 1]["id"]}
        for i in range(len(sorted_posts) - 1)
    ]
    return {"nodes": nodes, "edges": edges}

@app.get("/cluster-narratives")
def cluster_narratives():
    """
    Groups all collected posts (sample + scraped) into narrative clusters,
    showing which posts are likely spreading the same underlying story.
    """
    return cluster_posts(COLLECTED_POSTS)


@app.post("/generate-report")
def generate_report(text: str = Form(...)):
    """
    Runs full analysis on the input text and returns a human-readable
    investigation report alongside the structured data.
    """
    analysis = analyze_text(text=text)
    report_text = generate_investigation_report(analysis)
    return {
        "analysis": analysis,
        "report": report_text
    }