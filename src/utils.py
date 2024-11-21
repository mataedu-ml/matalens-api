import re
import os
import json
import base64
from typing import Dict, List
from logging import Logger
from template.question_format import Question
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain_chroma import Chroma

# LangSmith 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Mata-Lens"

def load_image_base64(image_path, logger)->str:
    """
    사진 로딩
    Args:
        image_path(string): 사진이 저장되어 있는 Path
        logger(Logger): FastAPI에 로깅
    """
    try:
        with open(image_path, "rb") as image_file:
            image = image_file.read()
        logger.info("Image file successfully loaded")
    except Exception as e:
        logger.error("Image file not found")
        logger.error(f"Detail: {e}")
        raise FileNotFoundError("Question image file is missing.")
    return base64.b64encode(image).decode('utf-8')

def math_problem_ocr(base64_image, logger)->Dict:
    """
    문제 OCR
    Args:
        base64_image(string): base64로 변환된 OCR 대상인 문제 이미지
        logger (Logger): FastAPI에 로깅
    """
    client = OpenAI()
    with open("prompts/ocr-prompt.txt", 'r') as prompt_file:
        prompt = prompt_file.read()

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
        logger.error("Problem with OCR using GPT")
        logger.error(f"Detail: {e}")
        raise ConnectionError("Problem with OCR using GPT")

    try:
        preprocessed = re.sub(r"(?<!\\)\\(?!\\)", r"\\\\", response.choices[0].message.content)
        question_dict = json.loads(preprocessed)
        logger.info("Question parsed correctly")
        logger.info(f"Original: \n {response.choices[0].message.content}")
        logger.info(f"Preprocessed: \n {preprocessed}")
    except Exception as e:
        logger.error("Responded result cannot be parsed")
        logger.error(f"Detail: {e}")
        logger.error(f"Detail: {response.choices[0].message.content}")
        # logging.info("Trying again...")
        raise TypeError("Responded result cannot be parsed into a dictionary format")

    return question_dict

def image_process(image_paths: List[str], logger: Logger) -> List[Question]:
    question_list = []
    for img_path in image_paths:
        # memo: 이미지를 불러오고 base64로 변환
        image_base64 = load_image_base64(img_path, logger)
        # memo: 이미지에서 텍스트 및 내용 추출
        question_dict = math_problem_ocr(image_base64, logger)
        # memo: 이미지 경로 정보 추가
        question_dict["img_path"] = img_path
        # memo: Question 객체 생성
        question_list.append(Question(**question_dict))
    return question_list

# question_text, 문제 텍스트 쿼리
def query_text(query_text, n_results=3):
    
    # OpenAI embeddings 초기화
    embeddings = OpenAIEmbeddings()
    
    # ChromaDB에 연결
    vectordb = Chroma(
        persist_directory="./chroma_db",
        collection_name="text_problems",
        embedding_function=embeddings
    )
    
    # 유사도 검색 실행
    results = vectordb.similarity_search(
        query_text,
        k=n_results
    )
    
    # Document 객체들의 id만 추출하여 리스트로 변환
    doc_ids = [doc.metadata['id'] for doc in results]
    
    # 중복 제거
    result = list(set(doc_ids))
    
    return result

# graph_or_chart, 이미지 해석 텍스트 쿼리
def query_image(graph_or_chart, n_results=3):
    
    # OpenAI embeddings 초기화
    embeddings = OpenCLIPEmbeddings()

    # ChromaDB에 연결
    vectordb = Chroma(
        persist_directory="./chroma_db",
        collection_name="text_to_image",
        embedding_function=embeddings        
    )
    
    # 유사도 검색 실행
    results = vectordb.similarity_search(
        graph_or_chart,
        k=n_results
    )
    
    # Document 객체들의 id만 추출하여 리스트로 변환
    doc_ids = [doc.metadata['id'].split('_')[1].split('.')[0] for doc in results]

    # 중복 제거
    result = list(set(doc_ids))
    
    return result

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
                        tag = line[tag_start+1:tag_end]
                        if tag.startswith('w'):
                            tags.append(tag)
    
    # 중복 제거 및 정렬
    return sorted(list(set(tags)))


def concept_explanation_response(message):
    try:
        llm = ChatOpenAI(model="gpt-4o",
                        top_p=1.0,
                        temperature=0)
        
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
        
        # 문제 자체로 검색
        problem_docs = vectordb.similarity_search(message, k=3)
        all_relevant_docs = problem_docs[1:]
        first_relevant_docs = [problem_docs[0]]
        
        # 키워드 추출 후 개별 검색
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
        
        return context
            
    except Exception as e:
        if "API" in str(e):
            return "API 연결에 문제가 발생했습니다. 잠시 후 다시 시도해주세요."
        elif "embedding" in str(e).lower():
            return "벡터 데이터베이스 검색 중 오류가 발생했습니다."
        else:
            return f"예상치 못한 오류가 발생했습니다: {str(e)}"

def auto_tagging(questions: List[Question], logger: Logger)->Dict:
    """
    문제 자동 태깅
    Args:
        question(Question): 태깅 대상인 문제

    Input data:
        question_text: 문제 텍스트
        graph_or_chart: 그래프, 차트 해석
        
    Returns:
        List[Dict]: 각 문제별 태깅 결과 리스트
    """
    
    results = []
    
    for i, question in enumerate(questions):
        question_result = {
            "id": i + 1,
            "concept_ids": [],
            "question_ids": []
        }
        
        # question_ids 업데이트
        question_result["question_ids"].extend(query_text(question.question_text))
        if question.graph_or_chart:  # None이 아닐 때만 실행
            question_result["question_ids"].extend(query_image(question.graph_or_chart))
            
        # concept_ids 업데이트
        tagging = concept_explanation_response(question.question_text)
        question_result["concept_ids"].extend(extract_tags(tagging))
        
        # 중복 제거
        question_result["concept_ids"] = list(set(question_result["concept_ids"]))
        question_result["question_ids"] = list(set(question_result["question_ids"]))
        
        results.append(question_result)
    
    return results


def question_analysis(image_paths, logger):
    question_list = image_process(image_paths, logger)
    tagging_list = auto_tagging(question_list, logger)
    return tagging_list

