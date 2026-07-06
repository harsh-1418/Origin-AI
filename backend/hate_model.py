from transformers import pipeline

# Real pretrained hate speech / toxicity model — no training needed.
# Loads once at startup, reused across requests.
_classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert",
    top_k=None  # return all label scores
)

def hate_speech_score(text: str) -> dict:
    """
    Returns a dict of toxicity-related scores from a real transformer model.
    Labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
    """
    results = _classifier(text[:512])[0]  # truncate to model's max length
    scores = {r["label"]: round(r["score"], 3) for r in results}

    # Overall score = max of the harmful categories
    harmful_labels = ["toxic", "severe_toxic", "threat", "identity_hate", "insult"]
    overall = max(scores.get(l, 0) for l in harmful_labels)

    return {
        "overall_score": round(overall, 3),
        "breakdown": scores
    }

def classify_threat(overall_score: float) -> str:
    if overall_score >= 0.7:
        return "HIGH"
    elif overall_score >= 0.4:
        return "MEDIUM"
    return "LOW"