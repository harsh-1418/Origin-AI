import os
import requests
import time
from datetime import datetime, timezone


def scrape_reddit(query: str, limit: int = 25, retries: int = 2):
    url = "https://www.reddit.com/search.json"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    }
    params = {"q": query, "sort": "new", "limit": limit}

    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries:
                time.sleep(2)
            else:
                return []

    posts = []
    for child in data.get("data", {}).get("children", []):
        p = child["data"]
        posts.append({
            "id": p["id"],
            "platform": "Reddit",
            "user": p.get("author", "unknown"),
            "text": (p.get("title", "") + " " + p.get("selftext", "")).strip(),
            "timestamp": datetime.fromtimestamp(
                p["created_utc"], tz=timezone.utc
            ).isoformat(),
            "url": f"https://reddit.com{p.get('permalink', '')}"
        })
    return posts


# ---------------- YOUTUBE (still needs a free key, no approval wall) ----------------

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

def scrape_youtube_comments(video_id: str, max_results: int = 25):
    if not YOUTUBE_API_KEY:
        return {"error": "YOUTUBE_API_KEY not set in .env"}

    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    comments = []
    for item in data.get("items", []):
        c = item["snippet"]["topLevelComment"]["snippet"]
        comments.append({
            "id": item["id"],
            "platform": "YouTube",
            "user": c.get("authorDisplayName", "unknown"),
            "text": c.get("textDisplay", ""),
            "timestamp": c.get("publishedAt", "")
        })
    return comments