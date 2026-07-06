def generate_investigation_report(analysis_result: dict) -> str:
    """
    Turns a structured analysis result into a readable investigation
    summary paragraph — template-based, no external LLM call needed,
    so it works instantly and offline.
    """
    threat = analysis_result.get("threat_level", "UNKNOWN")
    hate = analysis_result.get("hate_speech", {}).get("overall_score", 0)
    lang = analysis_result.get("language_detected", "unknown")
    evidence_count = analysis_result.get("evidence_count", 0)
    originator = analysis_result.get("likely_originator")

    lines = []
    lines.append(f"Threat Assessment: This content has been classified as {threat} risk, "
                  f"with a toxicity/hate-speech confidence score of {hate}.")
    lines.append(f"Detected language: {lang.upper()}.")

    if evidence_count > 0:
        lines.append(f"{evidence_count} related post(s) were found across the monitored dataset, "
                      f"indicating this narrative has propagated beyond a single source.")
    else:
        lines.append("No closely related posts were found in the current dataset, "
                      "suggesting this may be an isolated or newly emerging post.")

    if originator:
        lines.append(
            f"Based on timestamp ordering and semantic similarity across matched posts, "
            f"the most probable originator is '{originator['user']}' on {originator['platform']}, "
            f"first seen at {originator['timestamp']}, with a confidence score of "
            f"{originator['confidence']}."
        )
        lines.append("This is a likelihood estimate, not a confirmed identification, "
                      "and is recommended for further manual investigator review.")
    else:
        lines.append("Insufficient evidence was available to estimate a likely originator.")

    return " ".join(lines)