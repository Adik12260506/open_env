# server/env.py
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import EmailObservation, EmailAction
from server.tasks import load_emails, grade_classification, grade_reply, grade_summarize


class EmailEnv(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self.inbox = []
        self.history = []
        self._step_count = 0
        self._episode_id = "email_episode_001"

    def reset(self) -> EmailObservation:
        self.inbox = load_emails()
        self.history = []
        self._step_count = 0
        return EmailObservation(
            inbox=[e["text"] for e in self.inbox[1:]],
            current_email=self.inbox[0]["text"],
            history=self.history,
            reward=0.0,
            done=False,
        )

    def step(self, action: EmailAction) -> EmailObservation:
        if not self.inbox:
            return EmailObservation(
                inbox=[], current_email="", history=self.history,
                reward=0.0, done=True,
            )

        email = self.inbox.pop(0)
        self._step_count += 1
        action_type = (action.action_type or "classify").lower().strip()
        content = action.content or ""

        if action_type == "classify":
            raw = grade_classification(email, content)
        elif action_type == "reply":
            raw = grade_reply(email, content)
        elif action_type == "summarize":
            raw = grade_summarize(email, content)
        else:
            raw = 0.0

        reward = round(0.80 + raw * 0.20, 3)
        self.history.append(f"{action_type}: {content[:60]}")
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
        return State(episode_id=self._episode_id, step_count=self._step_count)

    def close(self):
        pass