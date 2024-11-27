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
from openai import OpenAI, AsyncOpenAI
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


# question_text, 문제 텍스트 쿼리
def query_text(query_text, n_results=3, logger: Logger = None):
    try:
        # OpenAI embeddings 초기화
        embeddings = OpenAIEmbeddings()
        if logger:
            logger.info("텍스트 임베딩 초기화 완료")

        # ChromaDB에 연결
        vectordb = Chroma(
            persist_directory="./chroma_db",
            collection_name="text_problems",
            embedding_function=embeddings
        )
        if logger:
            logger.info("ChromaDB 연결 완료")

        # 유사도 검색 실행
        results = vectordb.similarity_search(query_text, k=n_results)
        if logger:
            logger.info(f"텍스트 유사도 검색 완료: {len(results)}개 결과 찾음")

        # Document 객체들의 id만 추출하여 리스트로 변환
        doc_ids = [doc.metadata['id'] for doc in results]
        result = list(set(doc_ids))

        return result
    except Exception as e:
        if logger:
            logger.error(f"텍스트 쿼리 중 오류 발생: {str(e)}")
        raise


def query_image(graph_or_chart, n_results=3, logger: Logger = None):
    try:
        if logger:
            logger.info("이미지 임베딩 초기화 시작")
        # OpenAI embeddings 초기화
        embeddings = OpenCLIPEmbeddings()
        if logger:
            logger.info("이미지 임베딩 초기화 완료")

        # ChromaDB에 연결
        vectordb = Chroma(
            persist_directory="./chroma_db",
            collection_name="text_to_image",
            embedding_function=embeddings
        )
        if logger:
            logger.info("ChromaDB 연결 완료")

        # 유사도 검색 실행
        results = vectordb.similarity_search(graph_or_chart, k=n_results)
        if logger:
            logger.info(f"이미지 유사도 검색 완료: {len(results)}개 결과 찾음")

        doc_ids = [doc.metadata['id'].split('_')[1].split('.')[0] for doc in results]
        result = list(set(doc_ids))

        return result
    except Exception as e:
        if logger:
            logger.error(f"이미지 쿼리 중 오류 발생: {str(e)}")
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


def extract_tags(tagging):
    # 모든 'w'로 시작하는 태그를 추출
    tags = []
    sections = tagging.split('개념:')

    for section in sections:
        if '(w' in section:
            # 각 줄에서 태그 추출
            for line in section.split('\n'):
                if '(w' in line:
                    tag_start = line.find('(w')
                    tag_end = line.find(')', tag_start)
                    if tag_end != -1:
                        tag = line[tag_start + 1:tag_end]
                        if tag.startswith('w'):
                            tags.append(tag)

    # 중복 제거 및 정렬
    return sorted(list(set(tags)))


def concept_explanation_response(message, logger: Logger = None):
    try:
        if logger:
            logger.info("개념 설명 응답 시작")

        llm = ChatOpenAI(model="gpt-4o", top_p=1.0, temperature=0)
        if logger:
            logger.info("ChatOpenAI 모델 초기화 완료")

        with open("prompts/concept_explanation.txt", 'r') as prompt_file:
            keyword_prompt = prompt_file.read()

        keyword_messages = [
            SystemMessage(content=keyword_prompt),
            HumanMessage(content=message)
        ]

        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(persist_directory="./chroma_db",
                          collection_name="langchain",
                          embedding_function=embeddings)

        all_relevant_docs = []
        first_relevant_docs = []

        # 키워드 추출
        keywords = llm.invoke(keyword_messages).content
        if logger:
            logger.info(f"키워드 추출 완료: {keywords}")

        # 문제 자체로 검색
        problem_docs = vectordb.similarity_search(message, k=3)
        if logger:
            logger.info(f"문제 관련 문서 검색 완료: {len(problem_docs)}개")
        all_relevant_docs = problem_docs[1:]
        first_relevant_docs = [problem_docs[0]]

        # 키워드 추출 후 개별 검색
        if isinstance(keywords, str):
            keywords_list = keywords.split(',')

            # 각 키워드별로 개별 검색 수행
            for keyword in keywords_list:
                docs = vectordb.similarity_search(keyword.strip(), k=3)
                all_relevant_docs.extend(docs[1:])
                first_relevant_docs.extend([docs[0]])

            # 모든 문서 합치기
            all_docs = first_relevant_docs + all_relevant_docs

            # 중복 제거
            unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
            # 상위 5개로 제한
            context = format_docs(unique_docs)

            if logger:
                logger.info(f"총 {len(unique_docs)}개의 관련 문서 찾음")

            return context
        else:
            raise ValueError("Keywords extraction failed")

    except Exception as e:
        if logger:
            logger.error(f"개념 설명 응답 중 오류 발생: {str(e)}")
        if "API" in str(e):
            return "API 연결에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
        elif "embedding" in str(e).lower():
            return "벡터 데이터베이스 검색 중 오류가 발생했습니다."
        else:
            return f"예상치 못한 오류가 발생했습니다: {str(e)}"


def auto_tagging(questions, logger: Logger) -> List[Dict]:
    """
    문제 자동 태깅
    Args:
        question(Question): 태깅 대상인 문제
        logger(Logger): 로깅을 위한 Logger 객체

    Input data:
        question_text: 문제 텍스트
        graph_or_chart: 그래프, 차트 해석
        
    Returns:
        List[Dict]: 각 문제별 태깅 결과 리스트
    """

    results = []

    for i, question in enumerate(questions):
        question_result = {
            # "id": i + 1,
            "img_path": question.image_path,
            "concept_ids": [],
            "question_ids": []
        }

        # question_ids 업데이트 - logger 전달
        question_result["question_ids"].extend(query_text(question.question_text, logger=logger))
        # if question.graph_or_chart:  # None이 아닐 때만 실행
        #     question_result["question_ids"].extend(query_image(question.graph_or_chart, logger=logger))

        # concept_ids 업데이트 - logger 전달
        tagging = concept_explanation_response(question.question_text, logger=logger)
        question_result["concept_ids"].extend(extract_tags(tagging))

        # 중복 제거
        question_result["concept_ids"] = list(set(question_result["concept_ids"]))
        question_result["question_ids"] = list(set(question_result["question_ids"]))

        results.append(question_result)

    return results


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
    # 자동 태깅 시작
    tagging_list = auto_tagging(question_list, logger)
    return tagging_list
