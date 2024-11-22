# Python 3.10-slim 기반 이미지
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /code

# 필요한 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 최신 pip 업그레이드
RUN pip install --upgrade pip

# PyTorch 설치 (CPU 전용)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Python 의존성 설치
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

# FastAPI 서비스 포트 노출
EXPOSE 8000

# 애플리케이션 코드 복사
COPY . .

# FastAPI 실행 명령어
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8000"]
