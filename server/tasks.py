# def load_emails():
#     return [
#         {"text": "Refund request for damaged product", "type": "important"},
#         {"text": "Meeting reschedule", "type": "important"},
#         {"text": "You won a lottery!!!", "type": "spam"},
#         {"text": "Angry complaint about delay", "type": "important"},
#     ]


# def grade_classification(email, prediction):
#     if email["type"] == prediction:
#         return 1.0
#     return 0.0


# def grade_reply(email, reply):
#     text = email["text"].lower()
#     reply = (reply or "").lower()

#     if "refund" in text and "refund" in reply:
#         return 1.0
#     if "complaint" in text and ("sorry" in reply or "apologize" in reply):
#         return 1.0
#     return 0.0



# def load_emails():
#     return [
#         {"text": "Refund request for damaged product", "type": "important"},
#         {"text": "Meeting reschedule", "type": "important"},
#         {"text": "You won a lottery!!!", "type": "spam"},
#         {"text": "Angry complaint about delay", "type": "important"},
#     ]

# def load_emails(task_level="easy"):
    
#     if task_level == "easy":
#         return [
#             {"text": "Refund request", "type": "important"},
#             {"text": "Lottery win!!!", "type": "spam"},
#         ]

#     elif task_level == "medium":
#         return [
#             {"text": "Refund for damaged product", "type": "important"},
#             {"text": "Meeting reschedule", "type": "important"},
#             {"text": "Spam promotion offer", "type": "spam"},
#         ]

#     elif task_level == "hard":
#         return [
#             {"text": "Angry customer complaining about delay", "type": "important"},
#             {"text": "Refund request with frustration", "type": "important"},
#             {"text": "Fake lottery scam message", "type": "spam"},
#         ]

# def grade_classification(email, prediction):
#     if email["type"] == prediction:
#         return 1.0
#     return 0.0


# def grade_reply(email, reply):
#     text = email["text"].lower()
#     reply = (reply or "").lower()

#     if "refund" in text and "refund" in reply:
#         return 1.0
#     if "complaint" in text and ("sorry" in reply or "apologize" in reply):
#         return 1.0

#     return 0.0



# server/tasks.py
def load_emails():
    return [
        {"text": "Refund request", "type": "important"},
        {"text": "Lottery win!!!", "type": "spam"},
    ]


def grade_classification(email, prediction):
    prediction = str(prediction).lower().strip()
    if "spam" in prediction and "spam" in email["type"].lower():
        return 1.0
    if "important" in prediction and "important" in email["type"].lower():
        return 1.0
    return 0.6


def grade_reply(email, reply):
    reply_lower = str(reply or "").lower()
    if "refund" in email["text"].lower():
        if any(w in reply_lower for w in ["refund", "apologize", "sorry", "processed", "credited"]):
            return 1.0
        return 0.85   # good partial score
    return 0.7