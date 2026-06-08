# Wikipedia TTS Prototype (wiki-tts)

A Text-to-Speech (TTS) prototype that fetches Wikipedia articles, cleans the text (removes citation brackets, edit tags, etc), splits content by section, and generates .mp3 files asynchronously. 

## Architecture

Below is how the prototype works on Toolforge:
![Wikipedia TTS prototype architecture](assets/Wikipedia%20TTS%20prototype%20architecture.jpg)

## Toolforge Deployment

### 1. SSH into Toolforge and clone the repo
Log into the Toolforge, assume the tool account, and pull the source code.
```bash
$ ssh YOUR_USERNAME@login.toolforge.org
$ become wiki-tts

# Create the web directory and clone the repo
$ mkdir -p ~/www/python/src
$ git clone https://gitlab.wikimedia.org/toolforge-repos/wiki-tts.git ~/www/python/src
$ cd ~/www/python/src
```

### 2. Install dependencies

#### 2.1. FFmpeg (system dependency)
To compress audio to `.mp3` in a restricted Toolforge Linux environment without `sudo`, we use a static binary for FFmpeg.
```bash
# Navigate to local bin (create it if missing)
$ mkdir -p ~/bin && cd ~/bin

# Download the latest FFmpeg amd64 static build
$ wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz

# Unpack the archive and move binaries to ~/bin
$ tar xvf ffmpeg-release-amd64-static.tar.xz
$ cp ffmpeg-*-amd64-static/ffmpeg ~/bin/
$ cp ffmpeg-*-amd64-static/ffprobe ~/bin/

# Make them executable and clean up
$ chmod +x ~/bin/ffmpeg ~/bin/ffprobe
$ rm -rf ffmpeg-*-amd64-static*

# Confirm successful installation
$ ~/bin/ffmpeg -version
```

#### 2.2. Python environment
Build the virtual environment inside the Toolforge web shell to ensure C-extensions (like ONNX and hiredis) compile against the correct target OS.
```bash
$ toolforge webservice python3.11 shell
$ cd ~/www/python/src
$ python3 -m venv ~/www/python/venv
$ source ~/www/python/venv/bin/activate

# Install dependencies from requirements
$ pip install --upgrade pip wheel
$ pip install -r requirements.txt

# Download the Kokoro TTS model, voice profiles, and Wav2Vec2 alignment model directly into the expected paths.
$ python3 scripts/download_models.py

# Pre-compile the NeMo grammar cache so the celery worker starts faster.
# This avoids the k8s pod startup penalty (CrashLoopBackOff) from the 60+ second compilation timeout.
$ python3 scripts/initialize_nemo_cache.py
$ exit
```

### 3. Redis Message Broker
The Celery queue relies on [Toolforge's shared Redis](https://wikitech.wikimedia.org/wiki/Help:Toolforge/Redis). No setup is required. Connection is handled automatically via `redis.svc.tools.eqiad1.wikimedia.cloud:6379`.

### 4. Start Celery workers
Start the background inference workers as a continuous Toolforge job. This spans 10 replicas, pinned to 1 CPU thread each, to prevent CPU thrashing and ensure fast parallel generation. (see [T425804#11914308](https://phabricator.wikimedia.org/T425804#11914308) and [P93273](https://phabricator.wikimedia.org/P93273))
```bash
$ cd ~/www/python/src
$ toolforge jobs run celery-worker \
--command "export PYTHONUNBUFFERED=1 ORT_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 && cd ~/www/python/src && ~/www/python/venv/bin/celery -A wiki_tts.worker worker --concurrency=1 --max-tasks-per-child=5 --loglevel=info" \
--image python3.11 \
--continuous \
--replicas 5 \
--mem 4Gi \
--cpu 2

# Confirm the workers are running
$ toolforge jobs list
# Verify all replicas are running. This gives richer status than the jobs list (e.g OOM kills, CrashLoopBackOff, etc)
$ kubectl get pods
```

### 5. Start TTS service
Launch the FastAPI application using Toolforge's webservice router.
```bash
$ cd ~/www/python/src
$ toolforge webservice python3.11 start

# Confirm the web service is running
$ toolforge webservice status
```

### 6. Interacting with the service

#### 6.1. Programmatically via cURL
The primary endpoint that the Apps team will be hitting serves the .mp3 if it exists, or queues it and returns a `202 Accepted` status if missing.
```bash
# File exists returns HTTP 200 and downloads the .mp3 with the original filename.
$ curl -OJ "https://wiki-tts.toolforge.org/audio?article=Earth&section=Lead"

# File missing returns HTTP 202 instead of 404, queues section generation, and returns JSON status.
$ curl -w "\nHTTP Status: %{http_code}\n" "https://wiki-tts.toolforge.org/audio?article=Earth&section=Atmosphere"

# Section not found returns HTTP 404 with "Section 'FakeSection' not found".
$ curl -w "\nHTTP Status: %{http_code}\n" "https://wiki-tts.toolforge.org/audio?article=Earth&section=FakeSection"

# Article not found returns HTTP 404 with "Article 'NonExistent' not found".
$ curl -w "\nHTTP Status: %{http_code}\n" "https://wiki-tts.toolforge.org/audio?article=NonExistent&section=Lead"

# Generate missing sections of an article.
$ curl -X POST "https://wiki-tts.toolforge.org/generate?articles=Earth"

# Generate missing sections of multiple articles using a pipe-separated list just like the MediaWiki action API: https://en.wikipedia.org/w/api.php?action=query&prop=info&titles=Earth|Mars
$ curl -X POST "https://wiki-tts.toolforge.org/generate?articles=Earth|Mars"
```

#### 6.2. Visually via the browser
* **Prototype UI:** [https://wiki-tts.toolforge.org/](https://wiki-tts.toolforge.org/)
* **API Documentation:** [https://wiki-tts.toolforge.org/docs](https://wiki-tts.toolforge.org/docs)
* **Prototype Demo:** ![▶️ Watch the demo](assets/Wikipedia%20TTS%20prototype%20demo.mp4)

### 7. Batch TTS generation pipeline

#### 7.1. Extract featured article titles

Pull all Featured Article titles from the [categorymembers API](https://en.wikipedia.org/w/api.php?action=query&list=categorymembers&cmtitle=Category:Featured_articles&cmnamespace=0&cmlimit=max&format=json) and write them to a JSON file.

```bash
$ toolforge webservice python3.11 shell
$ cd ~/www/python/src
$ source ~/www/python/venv/bin/activate
$ python3 scripts/extract_featured_articles.py
$ less featured_articles.json
$ exit
```

#### 7.2. Submit articles for batch generation

The submission script reads `featured_articles.json` and POSTs batches of 25 pipe-separated titles to `/generate`. Before each batch it polls the Redis queue depth, and if the queue exceeds **200** pending tasks it backs off for **60 seconds**, then checks again. This auto-regulates throughput regardless of article-size variance.

The script also:
- Writes a checkpoint file (`submission_progress.json`) after every batch, so a restart resumes where it left off.
- Retries failed batches up to 3 times with exponential backoff, then writes them to `failed_articles.json` for later reprocessing.
- Responds to `SIGTERM`/`SIGINT` by finishing the current batch and saving progress before exiting.

**Dry-run first** to verify Redis connectivity without queueing any work:

> NB: delete old `submission_progress.json` and `failed_articles.json` if they are no longer needed before re-running.

```bash
$ rm ~/www/python/src/submission_progress.json ~/www/python/src/failed_articles.json

$ toolforge webservice python3.11 shell
$ cd ~/www/python/src
$ source ~/www/python/venv/bin/activate
$ python3 scripts/submit_articles.py --dry-run
$ exit
```

If the dry-run succeeds, launch the job:

```bash
$ cd ~/www/python/src
$ toolforge jobs run batch-generation \
--command "cd ~/www/python/src && ~/www/python/venv/bin/python3 scripts/submit_articles.py" \
--image python3.11 \
--continuous \
--replicas 1 \
--mem 512Mi \
--cpu 1

# Confirm all jobs are running (celery-worker, web server, and batch-generation)
$ toolforge jobs list

# Verify all pods are running. (celery workers, web server, and batch generation)
# This gives richer status than the jobs list (e.g OOM kills, CrashLoopBackOff, etc)
$ kubectl get pods
```

After all articles are submitted and the queue drains to zero, delete the job: `toolforge jobs delete batch-generation`

#### 7.3. Monitoring

Three things to watch while the pipeline runs:

| Metric | Command | Healthy threshold |
|---|---|---|
| Disk space | `df -h /data/project/` | > 1.0 TB free |
| Queue depth | `redis-cli -h redis.svc.tools.eqiad1.wikimedia.cloud LLEN wiki-tts-queue` | < 300 pending |
| Worker health | `kubectl get pods \| grep celery` | All 5 Running, 0 restarts |

#### 7.4. Retrying failed articles

If any articles fail after all retries, they are written to `failed_articles.json`. To retry them after the main run completes:

```bash
$ cd ~/www/python/src
$ toolforge jobs run batch-generation \
--command "cd ~/www/python/src && ~/www/python/venv/bin/python3 scripts/submit_articles.py --input failed_articles.json" \
--image python3.11 \
--continuous \
--replicas 1 \
--mem 512Mi \
--cpu 1
```

