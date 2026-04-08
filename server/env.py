# from models import Observation, Action
# from server.tasks import load_emails

# class EmailEnv:

#     def __init__(self):
#         self.inbox = []
#         self.current_index = 0
#         self.history = []

#     def reset(self):
#         self.inbox = load_emails()
#         self.current_index = 0
#         self.history = []

#         return Observation(
#             inbox=self.inbox,
#             current_email=self.inbox[0],
#             history=self.history
#         )

#     def step(self, action: Action):
#         email = self.inbox[self.current_index]

#         reward = 0.0
#         done = False

#         if action.action_type == "classify":
#             if "refund" in email.lower() and action.content == "important":
#                 reward += 0.5
#             elif "lottery" in email.lower() and action.content == "spam":
#                 reward += 0.5
#             else:
#                 reward -= 0.2

#         elif action.action_type == "reply":
#             if "refund" in email.lower() and "refund" in (action.content or "").lower():
#                 reward += 0.5
#             else:
#                 reward -= 0.2

#         self.history.append(f"{action.action_type}: {action.content}")

#         self.current_index += 1

#         if self.current_index >= len(self.inbox):
#             done = True
#             next_email = ""
#         else:
#             next_email = self.inbox[self.current_index]

#         return (
#             Observation(
#                 inbox=self.inbox,
#                 current_email=next_email,
#                 history=self.history
#             ),
#             reward,
#             done,
#             {}
#         )




# from server.tasks import load_emails, grade_classification, grade_reply
# from models import EmailObservation as Observation, EmailAction as Action
# class EmailEnv:

#     def __init__(self, task_level="easy"):
#         self.task_level = task_level
#         self.inbox = []
#         self.current_index = 0
#         self.history = []

#     def reset(self):
#         self.inbox = load_emails(self.task_level)
#         self.current_index = 0
#         self.history = []

#         return Observation(
#             inbox=[email["text"] for email in self.inbox],
#             current_email=self.inbox[0]["text"],
#             history=self.history
#         )

#     def step(self, action: Action):
#         email = self.inbox[self.current_index]

#         reward = 0.0
#         done = False

#         if action.action_type == "classify":
#             reward += grade_classification(email, action.content) * 0.5

#         elif action.action_type == "reply":
#             reward += grade_reply(email, action.content) * 0.5

#         else:
#             reward -= 0.1  # optional penalty

#         self.history.append(f"{action.action_type}: {action.content}")

#         self.current_index += 1

#         if self.current_index >= len(self.inbox):
#             done = True
#             next_email = ""
#         else:
#             next_email = self.inbox[self.current_index]["text"]

#         return (
#             Observation(
#                 inbox=[email["text"] for email in self.inbox],
#                 current_email=next_email,
#                 history=self.history
#             ),
#             reward,
#             done,
#             {}
#         )



# server/env.py
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from email_env.models import EmailObservation, EmailAction
from typing import Dict, Any


class EmailEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.inbox = []
        self.history = []
        self._step_count = 0
        self._episode_id = "email_episode_001"

    def reset(self) -> EmailObservation:
        self.inbox = [
            {"text": "Refund request", "type": "important"},
            {"text": "Lottery win!!!", "type": "spam"}
        ]
        self.history = []
        self._step_count = 0

        return EmailObservation(
            inbox=[e["text"] for e in self.inbox],
            current_email=self.inbox[0]["text"],
            history=self.history,
            reward=0.0,
            done=False,
        )

    def step(self, action: EmailAction) -> EmailObservation:
        if not self.inbox:
            return EmailObservation(
                inbox=[],
                current_email="",
                history=self.history,
                reward=0.0,
                done=True,
            )

        email = self.inbox.pop(0)
        self._step_count += 1

        reward = 1.0 if (action.action_type == "reply" and "refund" in email["text"].lower()) or \
                       (action.action_type == "classify" and "lottery" in email["text"].lower()) else 0.8

        self.history.append(f"{action.action_type}: {action.content or ''}")

        next_email = self.inbox[0]["text"] if self.inbox else ""
        done = len(self.inbox) == 0

        return EmailObservation(
            inbox=[e["text"] for e in self.inbox],
            current_email=next_email,
            history=self.history,
            reward=reward,
            done=done,
        )

    @property
    def state(self) -> State:
        """Return proper State object to fix 'episode_id' error"""
        return State(
            episode_id=self._episode_id,
            step_count=self._step_count
        )

    def close(self):
        pass