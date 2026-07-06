from transformers import pipeline

_classifier = None

def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            top_k=None
        )
    return _classifier

def hate_speech_score(text: str) -> dict:
    classifier = _get_classifier()
    results = classifier(text[:512])[0]
    scores = {r["label"]: round(r["score"], 3) for r in results}

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