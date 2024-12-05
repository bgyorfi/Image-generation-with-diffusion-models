FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./project/src ./src
COPY ./project/models ./models

COPY kaggle.json ./kaggle.json
RUN chmod 600 ./kaggle.json

RUN mkdir -p /app/datasets/flowers102 /app/datasets/celeba && \
    chmod -R 777 /app/datasets

CMD ["python", "src/main.py"]
