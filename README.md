# Wikipedia TTS Service

Text-to-Speech prototype for Wikipedia articles using Kokoro-82M via Kokoro-ONNX.
Built with **FastAPI**, **Celery**, **Redis**, and **Kokoro-ONNX**.

## Project Structure

```
wiki-tts/
├── app.py                          # uWSGI entry point (ASGIMiddleware wrapper)
├── uwsgi.ini                       # uWSGI configuration
├── requirements.txt                # Python dependencies
├── .gitignore
├── wiki_tts/                  # Main application package
│   ├── __init__.py
│   ├── config.py                   # Environment-based configuration
│   ├── text.py                     # Text normalization (numbers, citations, etc.)
│   ├── wikipedia_utils.py          # Wikipedia section traversal helpers
│   ├── locking.py                  # Redis distributed lock helpers
│   ├── routes.py                   # FastAPI app and API endpoints
│   ├── worker.py                   # Celery app and TTS generation task
│   └── static/
│       └── index.html              # Frontend UI
├── scripts/                        # (optional) Benchmark / utility scripts
├── tests/                          # (optional) Test suite
│   ├── __init__.py
│   └── test_text.py
└── audio_output/                   # Generated MP3 files
```

## Deployment (Toolforge)

### STEP 1: SSH into Toolforge and clone the repo

```shell
ssh YOUR_USERNAME@login.toolforge.org
become wiki-tts
git clone https://github.com/YOUR_ORG/wiki-tts.git ~/www/python/src
cd ~/www/python/src
```

### STEP 2: Install dependencies

#### 2.1 FFmpeg (static binary)

```shell
mkdir -p ~/bin && cd ~/bin
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xvf ffmpeg-release-amd64-static.tar.xz
cp ffmpeg-*-amd64-static/ffmpeg ~/bin/
cp ffmpeg-*-amd64-static/ffprobe ~/bin/
chmod +x ~/bin/ffmpeg ~/bin/ffprobe
rm -rf ffmpeg-*-amd64-static*
ffmpeg -version
```

#### 2.2 Python environment

```shell
toolforge webservice python3.11 shell
python3 -m venv ~/www/python/venv
source ~/www/python/venv/bin/activate
pip install --upgrade pip wheel
pip install -r ~/www/python/src/requirements.txt
exit
```

#### 2.3 Download the Kokoro model

Place `kokoro-v1.0.onnx` and `voices-v1.0.bin` in the repo root (`~/www/python/src/`).

### STEP 3: Start the Celery worker

```shell
toolforge jobs run celery-worker \
  --command "export ORT_NUM_THREADS=1 && cd ~/www/python/src && ~/www/python/venv/bin/celery -A wiki_tts.worker worker --pool solo --loglevel=info" \
  --image python3.11 \
  --continuous \
  --replicas 4 \
  --mem 2Gi \
  --cpu 1
```

> **Note:** If upgrading from the old flat structure, stop any old worker (`toolforge jobs stop celery-worker`) and start this one — the module path changed from `worker` to `wiki_tts.worker`.

### STEP 4: Start the web service

```shell
toolforge webservice python3.11 start
```

### STEP 5: Usage

```shell
# Download MP3 if it exists (HTTP 200)
curl -OJ "https://wiki-tts.toolforge.org/audio?article=Earth&section=Lead"

# Queue missing sections for one or more articles (HTTP 200)
curl -X POST "https://wiki-tts.toolforge.org/generate?articles=Earth|Mars"
```

Visit https://wiki-tts.toolforge.org/ in your browser.
API docs at https://wiki-tts.toolforge.org/docs.

## Local development

```shell
# Start Redis
redis-server

# Start the worker (in terminal 1)
celery -A wiki_tts.worker worker --pool solo --loglevel=info

# Start FastAPI (in terminal 2)
uvicorn wiki_tts.routes:app --reload --port 8000
```

## Tests

```shell
pip install pytest
pytest tests/
```
