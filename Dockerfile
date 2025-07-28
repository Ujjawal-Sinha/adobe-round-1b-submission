FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV SENTENCE_TRANSFORMERS_HOME=/app/models

RUN mkdir -p input output && \
    [ ! -f input/persona.txt ] && echo "Sample persona content." > input/persona.txt || true && \
    [ ! -f input/job_to_be_done.txt ] && echo "Sample job to be done content." > input/job_to_be_done.txt || true

CMD ["python", "main.py", "--input_dir", "input", "--output_dir", "output"]
