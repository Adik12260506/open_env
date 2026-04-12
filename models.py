from pydantic import BaseModel
from typing import List, Optional

class EmailObservation(BaseModel):
    inbox: List[str]
    current_email: str
    history: List[str]
    reward: float = 0.0
    done: bool = False

class EmailAction(BaseModel):
    action_type: str = "classify"
    content: Optional[str] = None