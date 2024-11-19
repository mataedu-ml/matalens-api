from typing import List
from pydantic import BaseModel


class Question(BaseModel):
    img_path: str
    question_count: int
    question_text: str
    is_multiple_choice: bool
    answer_choices: List[str] | None=None
    graph_or_chart: str | None=None


