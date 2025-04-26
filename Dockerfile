# Lightweight base; uses CPU OpenAI API so CUDA not mandatory.
FROM python:3.11-slim

WORKDIR /app

# Copy repo
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

CMD ["bash", "run.sh"]
