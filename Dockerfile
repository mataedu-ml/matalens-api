FROM python:3.11-alpine

WORKDIR /code
ENV FASTAPI_APP=service.py
ENV FASTAPI_RUN_HOST=0.0.0.0
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
COPY . .
CMD ["fastapi", "run", "--host=0.0.0.0", "--port=8000", "service.py"]

