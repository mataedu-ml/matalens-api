from typing import List
from pydantic import BaseModel


class Question(BaseModel):
    """
    Quetion 클래스 이미지 하나에 있는 문제(들)의 정보를 담은 객체
    - img_path: S3에 저장되어 있는 이미지의 경로
    - question_count: 이미지가 포함하고 있는 (온전한, 잘리지 않은)문제의 개수
    - question_text: 문제의 지문 텍스트
    - is_multiple_choice: 문제가 객관식인지, 혹은 주관식인지
    - answer_choices: 객관식 문제라면, 그 답 선택지
    """
    img_path: str
    question_count: int
    question_text: str
    is_multiple_choice: bool
    answer_choices: List[str] | None=None
    graph_or_chart: str | None=None


