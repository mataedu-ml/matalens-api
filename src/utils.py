import re
import os
import json
import base64
import boto3
import asyncio
import aiofiles
from time import time
from typing import Dict, List
from logging import Logger
from template.question_format import Question
from openai import OpenAI, AsyncOpenAI, OpenAIError
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma

# LangSmith 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Mata-Lens"


def load_image_base64_S3(image_path, logger) -> str:
    """
    S3와 통신해서 사진을 받아와서 base64로 변환해서 반환하는 메서드
    Args:
        image_path(string): 사진이 저장되어 있는 S3 Path
        logger(Logger): FastAPI에 로깅
    Returns:
        base64로 변환된 이미지 String
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


async def load_image_base64_local(image_path, logger) -> str:
    """
    로컬 경로 속 사진을 받아와서 base64로 변환해서 반환하는 메서드
    Args:
        image_path(string): 사진이 저장되어 있는 Local Path
        logger(Logger): FastAPI에 로깅
    Returns:
        base64로 변환된 이미지 String
    """

    try:
        async with aiofiles.open(image_path, "rb") as f:
            image_data = await f.read()
        logger.info(f"Image at {image_path} is successfully loaded")
    except Exception as e:
        logger.error(f"Image file not found at {image_path}")
        logger.error(f"Detail: {e}")
        raise FileNotFoundError("Question image file is missing.")

    return base64.b64encode(image_data).decode('utf-8')


async def ocr_single_image(image_path, prompt, logger) -> Question:
    """
    문제 속 텍스트 OCR
    Args:
        image_path(string): base64로 변환된 OCR 대상인 문제 이미지
        prompt(str): 이미지 속 텍스트를 추출하기 위해 API에 등록 할 프롬프트
        logger (Logger): FastAPI에 로깅
    Returns:
        문제를 파싱한 Question 객체
    """

    base64_image = await load_image_base64_local(image_path, logger)
    image_request = {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}",
        },
    }
    content = [{"type": "text", "text": prompt}, image_request]

    # memo: OpenAI API 호출
    client = AsyncOpenAI()

    # memo: API에 이미지 분석 요청 전송
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            max_tokens=1000
        )
        logger.info(f"API responded successfully for {image_path}")
    except ConnectionError as e:
        # memo: API 통신 실패시 오류 메세지 반환
        logger.error(f"Problem with OCR using GPT on {image_path}")
        logger.error(f"Detail: {e}")
        raise ConnectionError("Problem with OCR using GPT")

    try:
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"API response content is None for {image_path}")

        preprocessed = re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", content)
        question_dict = json.loads(preprocessed)
        question_dict["image_path"] = image_path
        logger.info("Question parsed with no error")
        logger.info(f"Result: \n {preprocessed}")
    except Exception as e:
        logger.error(f"Responded result from {image_path} cannot be parsed")
        logger.error(f"Detail: {e}")
        raise TypeError("Responded result cannot be parsed into a dictionary format")

    return Question(**question_dict)


async def ocr_multiple_images(image_paths: List[str], prompt: str, logger: Logger):
    """
    이미지 경로 리스트를 입력 받고, 모든 이미지에 대해서 OCR 작업을 해주는 메서드
    비동기로 이루어지기 때문에, 모든 이미지 프로세싱이 병렬로 이루어 진다.
    Args:
        image_paths: 각 이미지들이 저장되어 있는 경로들을 담고 있는 리스트
        prompt: 이미지에서 어떤 정보를 추출 할 지에 대한 설명이 적혀있는 프롬프트
        logger: FastAPI 로그
    Returns:
    """
    try:
        tasks = [ocr_single_image(image_path, prompt, logger) for image_path in image_paths]
        questions = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("All Questions parsed successfully")
    except Exception as e:
        raise Exception(f"Error at process_images_in_parallel {str(e)}")

    logger.info(f"Total Questions: {len(questions)}")

    return questions


async def query_text(questions: List[Question], vectordb, n_results, logger):
    """문제 텍스트 기반 유사도 검색"""
    try:
        logger.info("유사도 검색 시작")

        # 단일 문제인 경우 리스트로 변환
        if not isinstance(questions, list):
            questions = [questions]
            logger.debug("단일 문제를 리스트로 변환")

        # 모든 문제에 대해 동시에 검색 실행
        logger.info(f"총 {len(questions)}개 문제에 대한 검색 시작")
        search_tasks = [
            vectordb.asimilarity_search_with_relevance_scores(
                q.question_text if isinstance(q, Question) else q,
                k=n_results
            ) for q in questions
        ]
        search_results = await asyncio.gather(*search_tasks)
        logger.info("모든 문제 검색 완료")

        results = []

        for i, search_result in enumerate(search_results):
            # Document 객체들의 id와 유사도 점수 추출
            doc_ids = [doc.metadata['id'] for doc, score in search_result]
            scores = [score for doc, score in search_result]

            # 유사도 점수가 0.009 이상인 것만 필터링
            filtered = [(id, score) for id, score in zip(doc_ids, scores) if score > 0.009]
            logger.debug(f"문제 {i+1}: 필터링 후 {len(filtered)}개 결과")

            results.append({
                'ids': [id for id, _ in filtered],
                'scores': [score for _, score in filtered]
            })

        logger.info("유사도 검색 결과 처리 완료")
        return results

    except Exception as e:
        logger.error(f"유사도 검색 중 오류 발생: {str(e)}")
        raise


# format_docs 함수 추가
def format_docs(docs):
    if isinstance(docs, str):
        return docs
    return "\n\n".join([doc.page_content for doc in docs])


def format_docs_to_json(docs):
    data = []
    if isinstance(docs, str):
        # 문자열을 줄 단위로 분리하여 처리
        lines = docs.split('\n')
        info = {}
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                info[key.strip()] = value.strip()
        if info:  # info가 비어있지 않은 경우에만 추가
            data.append(info)
    else:
        # Document 객체 리스트 처리
        for doc in docs:
            info = {}
            lines = doc.page_content.split('\n')
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    info[key.strip()] = value.strip()
            if info:  # info가 비어있지 않은 경우에만 추가
                data.append(info)
    return data

async def extract_tags(tagging):
    """
    Document 객체 또는 Document 객체 리스트에서 'w'로 시작하는 태그를 추출

    Args:
        tagging: Document 객체 또는 {index: [Document]} 형태의 딕셔너리

    Returns:
        list: 추출된 태그 리스트
    """
    tags = []

    # 딕셔너리 형태로 입력된 경우
    if isinstance(tagging, dict):
        for docs_list in tagging.values():
            for doc in docs_list:
                if hasattr(doc, 'page_content'):
                    content = doc.page_content
                    if content.startswith('개념:'):
                        tag_start = content.find('(w')
                        if tag_start != -1:
                            tag_end = content.find(')', tag_start)
                            if tag_end != -1:
                                tag = content[tag_start+1:tag_end]
                                if tag.startswith('w'):
                                    tags.append(tag)

    # Document 객체 리스트인 경우
    elif isinstance(tagging, list) and all(hasattr(item, 'page_content') for item in tagging):
        for doc in tagging:
            content = doc.page_content
            if content.startswith('개념:'):
                tag_start = content.find('(w')
                if tag_start != -1:
                    tag_end = content.find(')', tag_start)
                    if tag_end != -1:
                        tag = content[tag_start+1:tag_end]
                        if tag.startswith('w'):
                            tags.append(tag)
    return tags

def get_text_vectordb(collection_name):
    embeddings = OpenAIEmbeddings()

    vectordb = Chroma(
        persist_directory="./chroma_db",
        collection_name=collection_name,
        embedding_function=embeddings
    )

    return vectordb

async def extract_keywords(message, logger: Logger):
    try:
        if logger:
            logger.info("개념 설명 응답 시작")

        llm = ChatOpenAI(model="gpt-4o", top_p=1.0, temperature=0)
        if logger:
            logger.info("ChatOpenAI 모델 초기화 완료")

        with open("prompts/concept_explanation.txt", 'r') as prompt_file:
            keyword_prompt = prompt_file.read()

        async def process_message(msg):
            # Question 객체에서 텍스트 추출 또는 문자열 그대로 사용
            content = str(msg.question_text if isinstance(msg, Question) else msg)

            messages = [
                SystemMessage(content=keyword_prompt),
                HumanMessage(content=content)
            ]

            response = await llm.ainvoke(messages)
            return response.content

        # 리스트 또는 단일 메시지 처리
        if isinstance(message, list):
            responses = await asyncio.gather(*[
                process_message(msg) for msg in message
            ])
            keywords = [resp for resp in responses]
        else:
            response = await process_message(message)
            keywords = [response]

        if logger:
            logger.info(f"추출된 키워드: {keywords}")

        return [key.split(', ') for key in keywords]

    except Exception as e:
        logger.error(f"개념 설명 생성 중 오류 발생: {str(e)}")

        if isinstance(e, OpenAIError):
            return "API 연결에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
        return f"예상치 못한 오류가 발생했습니다: {str(e)}"

def distribute_docs(tagging, keywords, num, tag_search_k):

    # 문제별 문서 리스트를 담을 Dict 초기화
    question_docs = {i: [] for i in range(num)}
    current_idx = num  # 시작 인덱스

    # 상위 num개 문서를 문제별로 우선 할당
    for i in range(num):
        if i < len(tagging):
            question_docs[i] = [tagging[i]]

    # 문제별로 키워드 기반 문서 할당
    for i in range(num):
        if i < len(keywords):
            # 현재 문제의 키워드에 대한 문서 수 계산
            tag_length = len(keywords[i]) * tag_search_k + tag_search_k
            end_idx = current_idx + tag_length

            # 인덱스 범위 체크
            if end_idx <= len(tagging):
                question_docs[i].extend(tagging[current_idx:end_idx])

            current_idx = end_idx

    return question_docs

async def search_with_tags(tagging, questions, vectordb, tag_search_k, logger):
    """태그와 문제 텍스트 기반 문서 검색 함수"""
    try:
        logger.info("태그 기반 문서 검색 시작")
        all_docs = []
        search_tasks = []

        # 1. 문제 텍스트 기반 검색 태스크 추가
        logger.debug(f"문제 텍스트 기반 검색 태스크 생성 중: {len(questions)}개 문제")
        for question in questions:
            question_text = question.question_text if hasattr(question, 'question_text') else question
            search_tasks.append(vectordb.asimilarity_search(question_text, k=1))

        # 2. 태그 기반 검색 태스크 추가
        logger.debug(f"태그 기반 검색 태스크 생성 중: {len(tagging)}개 태그 세트")
        for tag_set in tagging:
            # 태그 세트가 리스트인 경우 문자열로 변환
            if isinstance(tag_set, list):
                tag_set = ', '.join(tag_set)

            # 태그 세트 전체 검색 - k값 증가
            search_tasks.append(vectordb.asimilarity_search(tag_set, k=tag_search_k))

            # 개별 키워드 검색
            keywords = [kw.strip() for kw in tag_set.split(',')]
            for keyword in keywords:
                search_tasks.append(vectordb.asimilarity_search(keyword, k=tag_search_k))

        # 모든 검색 태스크 동시 실행
        logger.info(f"총 {len(search_tasks)}개의 검색 태스크 실행")
        search_results = await asyncio.gather(*search_tasks)

        # 검색 결과 처리
        for docs in search_results:
            all_docs.extend(docs)

        if logger:
            logger.info(f"총 {len(all_docs)}개의 고유 문서 검색됨")

        return all_docs

    except Exception as e:
        logger.error(f"문서 검색 중 오류 발생: {str(e)}")
        return []

async def auto_tagging(questions: List[Question], logger) -> List[Dict]:
    """
    문제 자동 태깅 함수

    Args:
        questions (List[Question]): 태깅할 문제 리스트
        logger: 로깅을 위한 Logger 객체
        
    Returns:
        List[Dict]: 각 문제별 태깅 결과 리스트.
                   각 딕셔너리는 img_path, concept_ids, question_ids를 포함
    """

    try:
        logger.info("자동 태깅 프로세스 시작")
        tag_search_k = 1
        num = len(questions)

        # 벡터 DB 초기화
        logger.info("벡터 DB 초기화")
        vectordb = get_text_vectordb("text_problems")
        vectordb_RAG = get_text_vectordb("langchain")

        # 비동기 작업 실행
        logger.info("비동기 작업 실행 시작")
        question_id = await query_text(questions, vectordb=vectordb, n_results=3, logger=logger)
        tagging = await extract_keywords(questions, logger=logger)
        tagging_docs = await search_with_tags(tagging, questions, vectordb=vectordb_RAG, tag_search_k=tag_search_k, logger=logger)
        concepts = await extract_tags(tagging_docs)
        concept_id = distribute_docs(concepts, tagging, num, tag_search_k)

        # 결과 조합
        logger.info("결과 조합 시작")
        results = []
        for i, q in enumerate(questions):
            question_result = {
                "img_path": q.image_path,
                "concept_ids": [],
                "question_ids": []
            }
            question_result["question_ids"].extend(question_id[i]["ids"])
            question_result["concept_ids"].extend(concept_id[i])
            results.append(question_result)

        logger.info(f"자동 태깅 완료: {len(results)}개 문제 처리됨")
        return results

    except Exception as e:
        logger.error(f"자동 태깅 중 오류 발생: {str(e)}")
        return []

async def question_analysis(image_paths, logger):
    """
    이미지 경로를 입력 받으면, 해당 이미지 속 문제들의 텍스트를 추출하여 개념을 자동 태깅을 하고,
    유사한 문제 Id를 반환하는 메서드

    Args:
        image_paths: 각 이미지들이 저장되어 있는 경로들을 담고 있는 리스트
        logger: FastAPI Logger

    Return:
        tagging_list: 각 이미지 별 태깅된 용어 리스트와
            "img_path": 텍스트를 추출한 대상 이미지의 경로
            "concept_ids": 이미지 속 문제와 연관된 용어 리스트
            "question_ids": 이미지 속 문제와 유사한 문제들의 id 리스트

    """
    # OCR 프롬프트 txt 가져오기
    async with aiofiles.open("prompts/ocr-prompt.txt", "r") as prompt_file:
        prompt = await prompt_file.read()
    # OCR 진행
    start_time = asyncio.get_event_loop().time()
    question_list = await ocr_multiple_images(image_paths, prompt, logger)
    question_list = [question for question in question_list if isinstance(question, Question)]
    finished_time = asyncio.get_event_loop().time() - start_time
    # OCR 완료 후 결과 반환
    logger.info(f"{question_list}")
    logger.info(f"Question text extraction took {finished_time} seconds")

    start_time = asyncio.get_event_loop().time()
    tagging_list = await auto_tagging(question_list, logger)
    finished_time = asyncio.get_event_loop().time() - start_time
    logger.info(f"Auto-tagging took {finished_time} seconds")

    return tagging_list
