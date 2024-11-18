import base64
from template.question import Question, Choice
from typing import Dict, List

def load_image(image_path)->str:
    """
    사진 로딩
    Args:
        image_path(string): 사진이 저장되어 있는 Path
    """
    try:
        with open(image_path, "rb") as image_file:
            image = image_file.read()
    except FileNotFoundError as e:
        print("File not found")
        print(f"Detail: {e}")
    return base64.b64encode(image_file.read()).decode('utf-8')

def math_problem_ocr(image)->Question:
    """
    문제 OCR
    Args:
        image(string): OCR 대상인 문제
    """
    question_dict = {
        "text": "다음 문제는 매우 문제이기 때문에, 이거에 대해서 문제 제기를 한 다음, 문제 해결을 해야하는 문제입니다.",
        "is_multiple_choice": True,
        "choices": [Choice(**{"number": n, "content": f"{n}"}) for n in range(1, 6)],
        "graph": "문제에 문제가 있는듯 한 그래프다."
    }
    question = Question(**question_dict)
    return question

def auto_tagging(question: Question)->List:
    """
    문제 자동 태깅
    Args:
        question(Question): 태깅 대상인 문제
    """
    return [choice.content for choice in question.choices]

def question_analysis(image):
    question_dict = math_problem_ocr(image)
    tagging_list = auto_tagging(question_dict)
    return tagging_list

