from typing import List
from pydantic import BaseModel


class Choice(BaseModel):
    number: int
    content: str


class Question(BaseModel):
    text: str
    is_multiple_choice: bool
    choices: List[Choice] | None=None
    graph: str | None=None


