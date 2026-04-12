# server/tasks.py

def load_emails():
    return [
        {
            "text": "I would like a refund for the damaged product I received last week. Order #12345.",
            "type": "important",
            "expected_action": "reply",
            "expected_keywords": ["refund", "apologize", "sorry", "processed", "credited"],
        },
        {
            "text": "Congratulations! You have won a $1,000,000 lottery prize. Click here to claim now!!!",
            "type": "spam",
            "expected_action": "classify",
            "expected_keywords": ["spam"],
        },
        {
            "text": "Can you please summarize the key points from our Q3 project meeting? I missed it.",
            "type": "important",
            "expected_action": "summarize",
            "expected_keywords": ["q3", "meeting", "project", "milestones", "key points", "discussed"],
        },
        {
            "text": "Angry complaint: My order has been delayed by 3 weeks and no one is responding!",
            "type": "important",
            "expected_action": "reply",
            "expected_keywords": ["sorry", "apologize", "delay", "update", "resolve"],
        },
    ]


def grade_classification(email: dict, prediction: str) -> float:
    prediction = str(prediction).lower().strip()
    expected = email.get("type", "").lower()
    return 1.0 if expected in prediction else 0.0


def grade_reply(email: dict, reply: str) -> float:
    reply_lower = str(reply or "").lower()
    keywords = email.get("expected_keywords", [])
    if not keywords:
        return 0.5
    matched = sum(1 for kw in keywords if kw in reply_lower)
    if matched >= 2:
        return 1.0
    return round(matched / len(keywords), 2)


def grade_summarize(email: dict, summary: str) -> float:
    summary_lower = str(summary or "").lower().strip()
    if not summary_lower or len(summary_lower) < 10:
        return 0.0
    keywords = email.get("expected_keywords", [])
    if not keywords:
        return 0.5
    matched = sum(1 for kw in keywords if kw in summary_lower)
    score = matched / len(keywords)
    if len(summary_lower) > 30 and matched >= 1:
        score = min(1.0, score + 0.2)
    return round(score, 2)