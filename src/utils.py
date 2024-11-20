import re
import json
import base64
import boto3
from typing import Dict, List
from openai import OpenAI
from logging import Logger
from template.question_format import Question

def load_image_base64(image_path, logger)->str:
    """
    사진 로딩
    Args:
        image_path(string): 사진이 저장되어 있는 Path
        logger(Logger): FastAPI에 로깅
    """
    try:
        s3 = boto3.client('s3')
        bucket_name = 'matalens-test-bucket'
        response = s3.get_object(Bucket=bucket_name, Key=image_path)
        image_data = response['Body'].read()

        logger.info(f"Image at {image_path} is successfully loaded")
    except Exception as e:
        logger.error("Image file not found")
        logger.error(f"Detail: {e}")
        raise FileNotFoundError("Question image file is missing.")
    return base64.b64encode(image_data).decode('utf-8')

def math_problem_ocr(base64_image, logger)->Dict:
    """
    문제 OCR
    Args:
        base64_image(string): base64로 변환된 OCR 대상인 문제 이미지
        logger (Logger): FastAPI에 로깅
    """

    # memo: OpenAI 클라이언트 생성
    client = OpenAI()

    # memo: API에 적용 할 프롬프트 호출
    with open("prompts/ocr-prompt.txt", 'r') as prompt_file:
        prompt = prompt_file.read()

    # memo: API에 이미지 분석 요청 전송
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        logger.info("API responded successfully")
    except ConnectionError as e:
        # memo: API 통신 실패시 오류 메세지 반환
        logger.error("Problem with OCR using GPT")
        logger.error(f"Detail: {e}")
        raise ConnectionError("Problem with OCR using GPT")

    # memo:
    try:
        preprocessed = re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", response.choices[0].message.content)
        question_dict = json.loads(preprocessed)
        logger.info("Question parsed with no error")
        # logger.info(f"Original: \n {response.choices[0].message.content}")
        logger.info(f"Result: \n {preprocessed}")
    except Exception as e:
        logger.error("Responded result cannot be parsed")
        logger.error(f"Detail: {e}")
        # logging.info("Trying again...")
        raise TypeError("Responded result cannot be parsed into a dictionary format")

    return question_dict

def image_process(image_paths: List[str], logger: Logger) -> List[Question]:
    question_list = []
    logger.info(f"===== Question Analysis Start "+"="*50)
    for img_path in image_paths:
        try:
            # memo: 이미지를 불러오고 base64로 변환
            image_base64 = load_image_base64(img_path, logger)
            # memo: 이미지에서 텍스트 및 내용 추출
            question_dict = math_problem_ocr(image_base64, logger)
            # memo: 이미지 경로 정보 추가
            question_dict["img_path"] = img_path
            # memo: Question 객체 생성
            question_list.append(Question(**question_dict))

            logger.info(f"Question {img_path} analysis is done!")
            logger.info("="*100)
        except Exception as e:
            continue

    return question_list

def auto_tagging(question: List[Question], logger: Logger)->Dict:
    """
    문제 자동 태깅
    Args:
        question(Question): 태깅 대상인 문제
    """

    result = {
        "concept_ids": [],
        "question_ids": [],
    }
    return result

def question_analysis(image_paths, logger):
    question_list = image_process(image_paths, logger)
    tagging_list = auto_tagging(question_list, logger)
    return tagging_list

