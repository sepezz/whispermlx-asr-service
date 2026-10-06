# Whispermlx ASR Service

[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)](https://github.com/KalebJS/whispermlx-asr-service/releases/tag/v0.4.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Platform: Apple Silicon](https://img.shields.io/badge/Platform-Apple%20Silicon%20%7C%20MLX-5856D6.svg)](https://github.com/ml-explore/mlx)
[![Python: 3.13](https://img.shields.io/badge/Python-3.13-3776AB.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-alpha-orange.svg)](https://github.com/KalebJS/whispermlx-asr-service)

**A native Apple-Silicon ASR API service powered by [whispermlx](https://pypi.org/project/whispermlx/) (MLX) with FastAPI.**

Runs natively on macOS with Apple Silicon (M1/M2/M3/M4). MLX Whisper inference runs on the Metal GPU automatically. No CUDA, no Docker, no Ray Serve.

## What This Does

- Transcribes audio files using OpenAI Whisper models via the MLX backend
- Identifies speakers ("Who spoke when") using Pyannote.audio
- Returns word-level timestamps via wav2vec2 alignment
- Supports 90+ languages
- Outputs JSON, SRT, VTT, TSV, and plain text formats
- OpenAI-compatible API (`/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/models`)
- Runs natively on Apple Silicon with `uv` and Python 3.13

## Limitations

- **Not production-grade**: Basic error handling, no authentication
- **Apple Silicon only**: Requires an M-series Mac. No NVIDIA/CUDA support.
- **File size limits**: Large audio files (>1GB) can cause out-of-memory errors
- **Memory usage**: RAM consumption increases with file size and diarization. Peak ~2.3 GB for a small-model full pipeline on a 16 GB M1.
- **Alpha software**: Expect bugs and breaking changes

## How It Works

```
Audio --> MLX Whisper (transcription, Metal GPU) --> Wav2Vec2 (alignment) --> Pyannote (speaker ID) --> Output
```

The service runs as a single-process uvicorn server with an async queue. Requests are serialized through a semaphore so only one pipeline runs on the Metal GPU at a time. This is suitable for single-device, low-traffic, or development use.

**Device semantics:** MLX Whisper ASR always runs on the Metal GPU automatically. The `DEVICE` environment variable (default `mps`) only controls where the VAD, wav2vec2 alignment, and pyannote diarization (torch-based stages) run. `COMPUTE_TYPE` and `BATCH_SIZE` are accepted for API compatibility but have no effect on the MLX backend.

For long podcasts on Macs with limited GPU memory, set `DEVICE=cpu` in the service environment. This moves speech detection and timestamp alignment to the CPU while keeping MLX Whisper transcription on the Metal GPU. Alignment and diarization may take longer. The service clears unused MLX and PyTorch GPU caches before and after transcription and alignment, including after a failed stage.

## Prerequisites

### Hardware Requirements

- **Apple Silicon Mac** (M1, M2, M3, or M4)
- **RAM:** 16 GB recommended (8 GB may work with `tiny`/`base` models)
- **Storage:** 50 GB SSD for model caching

Memory requirements vary by model size:

| Whisper Model | RAM (full pipeline*) | Notes |
|---------------|----------------------|-------|
| `tiny`, `base` | ~2 GB | Fast, low quality |
| `small` | ~2.3 GB | Good balance of speed and quality |
| `medium` | ~5 GB | Good quality, slower |
| `large-v3-turbo`, `turbo` | ~5 GB | Fast, high quality |
| `large-v3` | ~10+ GB | Best quality, slowest |

*Full pipeline = Whisper model + alignment model + pyannote speaker diarization. Measured on M1 16 GB.

### Software Requirements

- **macOS** with Apple Silicon
- **[uv](https://docs.astral.sh/uv/)** (Python package manager)
- **Python 3.13** (installed via uv; system Python 3.14 is incompatible with whispermlx)
- **FFmpeg** (for audio decoding; `brew install ffmpeg`)
- **Hugging Face Account** (for speaker diarization models)

## Quick Start

### 1. Install uv and Set Up Python 3.13

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/KalebJS/whispermlx-asr-service.git
cd whispermlx-asr-service

# Create a Python 3.13 virtual environment and install dependencies
uv venv --python 3.13
uv sync
```

### 2. Get Hugging Face Token (for Speaker Diarization)

Speaker diarization requires a Hugging Face token and model access:

**a) Create a Hugging Face Account:**
- Visit [https://huggingface.co/join](https://huggingface.co/join) and sign up

**b) Accept the Model User Agreement:**
- [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) - Click "Agree and access repository"

**c) Generate an Access Token:**
- Visit [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- Click "New token", name it (e.g., "whispermlx-diarization")
- Select "Read" permission and generate
- Copy the token (starts with `hf_...`)

> Without the token and accepted agreement, diarization is gracefully skipped and transcription still works, but no speaker labels will be assigned.

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your Hugging Face token
nano .env
```

Minimal `.env`:

```bash
HF_TOKEN=hf_your_token_here
DEVICE=mps
PRELOAD_MODEL=large-v3
PORT=9001
```

### 4. Run the Service

```bash
# Export your .env vars first (entrypoint.sh does NOT auto-load .env)
set -a; source .env; set +a

# Start the service (binds 0.0.0.0:9001)
./entrypoint.sh

# Or start directly with uvicorn (binds localhost only)
uv run uvicorn app.main:app --host 127.0.0.1 --port 9001

# Or load .env and start in one step
uv run uvicorn app.main:app --host 127.0.0.1 --port 9001 --env-file .env
```

The service will be available at `http://localhost:9001`.

> **Note:** `entrypoint.sh` hardcodes port 9001 and binds to `0.0.0.0` (all interfaces). The `PORT` env var is only respected when launching uvicorn directly with `--port $PORT`. Since the service has no authentication, prefer `--host 127.0.0.1` unless you need remote access.

> Port 9001 is the default. Port 9000 may be in use by other services (e.g., php-fpm on some macOS setups). The reserved port range for this service is 9001-9010.

### 5. Test the Service

```bash
# Health check
curl http://localhost:9001/health

# Test transcription
curl -X POST http://localhost:9001/asr \
  -F "audio_file=@your_audio.mp3" \
  -F "language=en"
```

A smoke test script is included:

```bash
./test-api.sh localhost 9001 path/to/audio.wav
```

---

## API Documentation

Once running, visit `http://localhost:9001/docs` for interactive API documentation.

### Main Endpoint: POST /asr

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `audio_file` | File | Required | Audio file to transcribe |
| `task` | String | `transcribe` | Task type: `transcribe` or `translate` |
| `language` | String | Auto-detect | Language code (e.g., `en`, `es`, `fr`) |
| `model` | String | `large-v3` | Whisper model (see [Model Selection](#model-selection)) |
| `initial_prompt` | String | None | Context or spelling guide to steer the model |
| `hotwords` | String | None | Accepted but **ignored** by the MLX backend (see [Hotwords](#hotwords-no-op)) |
| `output_format` | String | `json` | Output format: `json`, `text`, `srt`, `vtt`, `tsv` |
| `output` | String | None | Legacy alias for `output_format` |
| `word_timestamps` | Boolean | `true` | Return word-level timestamps |
| `diarize` | Boolean | `true` | Enable speaker diarization |
| `enable_diarization` | Boolean | None | Alias for `diarize` |
| `num_speakers` | Integer | Auto | Exact number of speakers (overrides min/max) |
| `min_speakers` | Integer | Auto | Minimum number of speakers |
| `max_speakers` | Integer | Auto | Maximum number of speakers |
| `return_speaker_embeddings` | Boolean | `false` | Return 256-dimensional speaker embedding vectors |

**Example Request (JSON output):**

```bash
curl -X POST http://localhost:9001/asr \
  -F "audio_file=@meeting.mp3" \
  -F "language=en" \
  -F "model=large-v3" \
  -F "output_format=json" \
  -F "diarize=true" \
  -F "min_speakers=2" \
  -F "max_speakers=5"
```

**Example Request (SRT subtitles):**

```bash
curl -X POST http://localhost:9001/asr \
  -F "audio_file=@video.mp4" \
  -F "language=en" \
  -F "output_format=srt" \
  -F "diarize=false"
```

**Example Response (JSON):**

The `text` field is a JSON array mirroring the `segments` array (legacy drop-in shape from the original whisper-asr-webservice):

```json
{
  "text": [
    {
      "start": 0.5,
      "end": 2.3,
      "text": " Hello, welcome to the meeting.",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Hello", "start": 0.5, "end": 0.8, "score": 0.95},
        {"word": "welcome", "start": 0.9, "end": 1.2, "score": 0.93}
      ]
    }
  ],
  "language": "en",
  "segments": [...],
  "word_segments": [...]
}
```

### Hotwords (No-Op)

The `hotwords` parameter is **accepted for API compatibility but is a no-op on the MLX backend**. The `whispermlx` library has no hotwords mechanism. When you supply `hotwords`, the service logs a warning (`"hotwords is ignored by the MLX backend"`) and proceeds with normal transcription. The parameter never causes an error.

To bias transcription toward specific spellings, use `initial_prompt` instead, which provides context that primes the model to expect certain terms:

```bash
# Use initial_prompt to guide spelling
curl -X POST "http://localhost:9001/asr?language=en&initial_prompt=Speakr+is+a+transcription+app." \
  -F "audio_file=@meeting.mp3"
```

### Speaker Diarization

Speaker diarization assigns `SPEAKER_NN` labels to segments and words. It is enabled by default when `HF_TOKEN` is set.

**Requirements:**
- `HF_TOKEN` must be set in your `.env` file
- You must have accepted the model agreement for [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

**When `HF_TOKEN` is missing or diarization fails:** The service gracefully skips diarization and returns the transcription without speaker labels (HTTP 200, no crash). This ensures transcription is never blocked by a missing token.

**Exact Speaker Count:**

```bash
curl -X POST http://localhost:9001/asr \
  -F "audio_file=@interview.mp3" \
  -F "num_speakers=2" \
  -F "diarize=true"
```

`num_speakers` overrides `min_speakers` and `max_speakers`.

**Speaker Embeddings:**

```bash
curl -X POST http://localhost:9001/asr \
  -F "audio_file=@meeting.mp3" \
  -F "return_speaker_embeddings=true" \
  -F "diarize=true"
```

Returns a `speaker_embeddings` object keyed by speaker label, with one numeric vector per detected speaker. Embeddings are only included in `json` output format.

### OpenAI-Compatible Endpoints

The service provides drop-in OpenAI API compatibility:

#### POST /v1/audio/transcriptions

```bash
curl -X POST http://localhost:9001/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=whisper-1"
```

Supports `response_format`: `json` (default, returns `{"text": "..."}`), `text`, `srt`, `vtt`, `verbose_json` (full object with segments, optional word timestamps via `timestamp_granularities[]`).

The `model` field accepts OpenAI-style aliases (`whisper-1`, `whisper-tiny`, `whisper-large-v3`) and raw MLX model names (`tiny`, `base`, `large-v3-turbo`, etc.).

#### POST /v1/audio/translations

```bash
curl -X POST http://localhost:9001/v1/audio/translations \
  -F "file=@spanish_audio.mp3" \
  -F "model=whisper-1"
```

Translates non-English audio into English text. Same response formats as transcriptions. `verbose_json` reports `task: "translate"`.

#### GET /v1/models

```bash
curl http://localhost:9001/v1/models
```

Returns an OpenAI-style list of available models, built from the MLX model map. Includes the `whisper-1` alias and all canonical MLX model names.

#### GET /v1/models/{model_id}

```bash
curl http://localhost:9001/v1/models/large-v3
```

Returns the matching model object, or a 404 OpenAI error for unknown ids.

### Health and Metrics

```bash
# Health check
curl http://localhost:9001/health
# {"status": "healthy", "device": "mps", "loaded_models": ["large-v3"], "serve_mode": "simple"}

# Root endpoint
curl http://localhost:9001/

# Prometheus metrics
curl http://localhost:9001/metrics

# Queue metrics (JSON)
curl http://localhost:9001/queue-metrics
```

**Prometheus Metrics:**

| Metric | Type | Notes |
|--------|------|-------|
| `whisperx_requests_total{endpoint,status}` | Counter | `status` is `ok`, `http_<code>`, or `error` |
| `whisperx_request_duration_seconds{endpoint}` | Histogram | End-to-end handler time |
| `whisperx_active_transcriptions` | Gauge | In-flight `/asr` requests |
| `whisperx_loaded_models` | Gauge | Whisper models currently in cache |
| `whisperx_model_evictions_total{model}` | Counter | Models unloaded by the idle-eviction sweep |
| `whisperx_audio_duration_seconds` | Histogram | Submitted audio duration |
| `whisperx_audio_size_megabytes` | Histogram | Submitted file size |
| `whisperx_vram_allocated_bytes` | Gauge | MLX active memory (or 0) |
| `whisperx_service_info` | Info | Static labels: version, device, compute_type, serve_mode |

The `whisperx_vram_allocated_bytes` gauge reports MLX active memory via `mlx.core.get_active_memory()` when available, or 0 otherwise. No `torch.cuda` is used.

---

## Model Selection

Available MLX Whisper models (speed vs accuracy tradeoff):

| Model | Parameters | Speed | Quality |
|-------|------------|-------|---------|
| `tiny`, `tiny.en` | 39M | Fastest | Lowest |
| `base`, `base.en` | 74M | Very Fast | Low |
| `small`, `small.en` | 244M | Fast | Medium |
| `medium`, `medium.en` | 769M | Moderate | Good |
| `large`, `large-v1` | 1550M | Slow | Excellent |
| `large-v2` | 1550M | Slow | Excellent |
| `large-v3` | 1550M | Slow | Best |
| `large-v3-turbo`, `turbo` | 809M | Fast | High |

OpenAI-style aliases (`whisper-1`, `whisper-tiny`, `whisper-large-v3`, etc.) are also accepted and resolve to the corresponding MLX model.

**Recommendation:**
- Use `large-v3` for best quality
- Use `small` or `base` for speed and lower memory usage
- Use `large-v3-turbo` for a good balance of speed and quality

Models are downloaded on first use and cached in `CACHE_DIR` (default `~/.cache/whisperx-asr`).

---

## Configuration

### Environment Variables

Edit `.env` to customize:

```bash
# Device for torch-based stages (VAD, alignment, diarization).
# MLX Whisper ASR always runs on the Metal GPU regardless of this setting.
DEVICE=mps              # mps (default, recommended) or cpu (fallback, slower diarization)

# Compute type and batch size (accepted but INERT under MLX — no effect on inference)
# Code defaults: COMPUTE_TYPE=int8, BATCH_SIZE=2 (leftover from CUDA era, unused)
#COMPUTE_TYPE=int8
#BATCH_SIZE=2

# Hugging Face token for diarization (REQUIRED for speaker labels)
HF_TOKEN=hf_xxx...

# Model preloading (optional, reduces first-request latency)
# Also sets the default model for /asr requests when no model= param is given
PRELOAD_MODEL=large-v3   # Leave empty to disable

# Override which model the OpenAI "whisper-1" alias resolves to
# (defaults to the PRELOAD_MODEL value, or large-v3 if unset)
#OPENAI_WHISPER1_MODEL=large-v3

# Service port (default 9001)
PORT=9001

# Model cache directories
CACHE_DIR=~/.cache/whisperx-asr
HF_HOME=~/.cache/whisperx-asr

# Maximum file size in MB (prevents out-of-memory errors)
MAX_FILE_SIZE_MB=1000

# GPU concurrency (Metal GPU semaphore; default 1)
#GPU_CONCURRENCY=1

# Maximum queued requests before rejecting with 503 (default 32)
#MAX_QUEUE_SIZE=32

# Idle model eviction (default disabled). When > 0, Whisper models that have
# not served a request in this many seconds are unloaded from memory.
#MODEL_KEEP_ALIVE_SECONDS=0
#MODEL_EVICTION_INTERVAL_SECONDS=60

# Offline mode (optional): set to 1 to prevent network requests after models are cached
#HF_HUB_OFFLINE=1
```

### Idle Model Eviction

Set `MODEL_KEEP_ALIVE_SECONDS` to unload Whisper models that have been idle longer than the configured window. The next request that needs the model reloads it transparently:

```bash
MODEL_KEEP_ALIVE_SECONDS=3600          # unload models idle for 1 hour
MODEL_EVICTION_INTERVAL_SECONDS=60     # sweep cadence (floor 30 seconds)
```

Default is `0` (disabled; models stay loaded).

---

## Integration with Speakr

To use this service with [Speakr](https://github.com/murtaza-nasir/speakr):

Update Speakr's `.env` file:

```bash
USE_ASR_ENDPOINT=true
ASR_BASE_URL=http://localhost:9001
```

If the service is on a different machine, replace `localhost` with the IP address and ensure the port is accessible through your firewall.

**If Speakr runs in Docker:** `localhost` inside a container refers to the container itself, not the host. Use `host.docker.internal` instead:

```bash
ASR_BASE_URL=http://host.docker.internal:9001
```

**Model compatibility:** `distil-*` models (e.g., `distil-large-v2`) are no longer available on the MLX backend. If Speakr was configured to use a `distil-*` model, switch to a standard model name such as `large-v3`, `small`, or `large-v3-turbo`. The `hotwords` parameter is accepted but silently ignored; use `initial_prompt` for spelling bias instead.

---

## Running the Service

```bash
# Using entrypoint.sh (exports .env first; binds 0.0.0.0:9001)
set -a; source .env; set +a
./entrypoint.sh

# Or directly with uvicorn (localhost only, auto-loads .env)
uv run uvicorn app.main:app --host 127.0.0.1 --port 9001 --env-file .env

# With environment variables inline (no .env needed)
DEVICE=mps PRELOAD_MODEL=base uv run uvicorn app.main:app --host 127.0.0.1 --port 9001
```

---

## Offline Use

The service can run completely offline after an initial setup with internet access:

1. Start the service with internet access
2. Run at least one transcription request with diarization enabled to cache all models:
   ```bash
   curl -X POST http://localhost:9001/asr \
     -F "audio_file=@test.mp3" \
     -F "diarize=true"
   ```
3. Set `HF_HUB_OFFLINE=1` in your `.env` file
4. Restart the service

The service will now operate without any network requests to Hugging Face.

---

## Monitoring and Logs

### View Logs

When running in the foreground, logs appear in the terminal. When running in the background:

```bash
# Export .env first, then start in the background
set -a; source .env; set +a
./entrypoint.sh &> service.log &
tail -f service.log
```

### Health Check

```bash
curl http://localhost:9001/health
# {"status": "healthy", "device": "mps", "loaded_models": ["large-v3"], "serve_mode": "simple"}
```

---

## Supported Audio Formats

The service supports formats decodable by FFmpeg:

- **Audio:** MP3, WAV, M4A, FLAC, AAC, OGG, WMA
- **Video:** MP4, AVI, MOV, MKV, WebM (audio track extracted)
- **Other:** AMR, 3GP, 3GPP

---

## Troubleshooting

### Speaker Diarization Not Working

**Symptom:** No speaker labels in output

**Solutions:**
1. Verify `HF_TOKEN` is set correctly in `.env`
2. Accept the model agreement at [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
3. Check logs for diarization errors
4. Ensure `diarize=true` in request (diarization defaults to true when `HF_TOKEN` is set)

> Without `HF_TOKEN`, diarization is silently skipped and transcription proceeds without speaker labels.

### Out of Memory Errors

**Solutions:**
1. Reduce `MAX_FILE_SIZE_MB` in `.env`
2. Use a smaller model (`small` or `base` instead of `large-v3`)
3. Disable diarization for very large files: `diarize=false`
4. Split large audio files into smaller chunks before uploading

### Slow Processing

**Solutions:**
1. Use a smaller model for faster processing
2. Disable diarization if not needed: `diarize=false`
3. Use `large-v3-turbo` for a good speed/quality balance

### API Returns 500 Errors

Check logs for error details. Common causes:
- Invalid audio format (use FFmpeg to convert)
- Model download failure (check internet access)
- Incorrect parameters (check API docs at `/docs`)

---

## Stress Testing

A stress test script is included to measure throughput and latency under concurrent load:

```bash
# Default: 4 concurrent workers, all files in testfiles/
uv run python tests/stress_test.py

# 8 concurrent workers, 3 rounds
uv run python tests/stress_test.py --workers 8 --rounds 3

# Test OpenAI-compat endpoint
uv run python tests/stress_test.py --endpoint openai

# Without diarization
uv run python tests/stress_test.py --no-diarize
```

Place audio files in the `tests/testfiles/` directory (gitignored).

---

## Security Notes

**This service has NO built-in authentication or security features.**

If exposing to a network:
- Use firewall rules to restrict access
- Consider putting behind a reverse proxy
- Store `HF_TOKEN` securely in the `.env` file (never hardcode)

---

## License

This project is MIT licensed. See [LICENSE](LICENSE) for details.

WhisperX is licensed under BSD-4-Clause. See [WhisperX repository](https://github.com/m-bain/whisperX) for details.

## Credits

- **whispermlx:** MLX fork of WhisperX for Apple Silicon
- **WhisperX:** [m-bain/whisperX](https://github.com/m-bain/whisperX)
- **OpenAI Whisper:** [openai/whisper](https://github.com/openai/whisper)
- **MLX:** [ml-explore/mlx](https://github.com/ml-explore/mlx)
- **Pyannote.audio:** [pyannote/pyannote-audio](https://github.com/pyannote/pyannote-audio)

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.
