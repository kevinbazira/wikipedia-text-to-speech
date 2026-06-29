FROM python:3.11-slim

WORKDIR /app

# System deps for scipy / numpy / onnxruntime C-extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY alignment.py inference.py server.py ./

# Models are mounted at /mnt/models by KServe (or bind-mounted for local dev)
ENV MODEL_DIR=/mnt/models
ENV ORT_NUM_THREADS=1
ENV OMP_NUM_THREADS=1

EXPOSE 8080

CMD ["python", "server.py"]
