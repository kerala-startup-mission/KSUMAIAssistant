import io
import base64
import os
import tempfile
import re
import shutil
import subprocess
import logging
from collections import deque
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Request, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.rag_engine import KSUMAssistant
import edge_tts
from faster_whisper import WhisperModel

# ==========================================
# LOGGING SETUP (Captures last 200 logs for Admin UI)
# ==========================================
LOG_BUFFER = deque(maxlen=200)

class RingBufferHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            LOG_BUFFER.append(msg)
        except Exception:
            pass

logger = logging.getLogger("ksum_assistant")
logger.setLevel(logging.INFO)
buffer_handler = RingBufferHandler()
buffer_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', '%H:%M:%S'))
logger.addHandler(buffer_handler)

# Also output to stdout
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', '%H:%M:%S'))
logger.addHandler(console_handler)

# ==========================================
# APP INITIALIZATION
# ==========================================
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="KSUM AI Voice Assistant API")

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("Initializing KSUM Assistant & STT Engine...")
assistant = KSUMAssistant()
whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
logger.info("KSUM Assistant System Ready.")

class ChatRequest(BaseModel):
    message: str

def clean_text_for_voice(text: str) -> str:
    clean = re.sub(r'[\*#`\-\_]', '', text)
    clean = clean.split("Source Documents:")[0].strip()
    return clean

# ==========================================
# PUBLIC CORE ENDPOINTS
# ==========================================
@app.get("/")
@limiter.limit("30/minute")
async def root(request: Request):
    return {"status": "online", "message": "KSUM AI Assistant API is running"}

@app.post("/chat")
@limiter.limit("10/minute")
async def chat_endpoint(request: Request, chat_data: ChatRequest):
    logger.info(f"Incoming Chat Request: {chat_data.message}")
    response_text = assistant.generate_response(chat_data.message)
    clean_text = clean_text_for_voice(response_text)
    audio_base64 = ""
    
    if clean_text:
        try:
            communicate = edge_tts.Communicate(clean_text, "en-US-JennyNeural")
            audio_buffer = io.BytesIO()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Audio generation failed: {e}")

    return {"response": response_text, "audio": audio_base64}

@app.post("/stt")
@limiter.limit("15/minute")
async def speech_to_text(request: Request, file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
            content = await file.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name
        
        segments, info = whisper_model.transcribe(
            temp_audio_path, 
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, min_speech_duration_ms=250),
            condition_on_previous_text=False,
            no_speech_threshold=0.5
        )
        
        transcript = "".join([s.text for s in segments if s.no_speech_prob < 0.6]).strip()
        os.remove(temp_audio_path)
        logger.info(f"STT Output: '{transcript}'")
        return {"transcript": transcript}
    except Exception as e:
        logger.error(f"STT Error: {e}")
        return {"transcript": "", "error": str(e)}

# ==========================================
# ADMIN & SYSTEM MANAGEMENT ENDPOINTS
# ==========================================
UPLOAD_DIR = Path("/app/data/core_website_data")

@app.get("/api/admin/logs")
@limiter.limit("60/minute")
async def get_system_logs(request: Request):
    """Returns the latest 200 system log messages."""
    return {"logs": list(LOG_BUFFER)}

@app.post("/api/admin/upload")
@limiter.limit("20/minute")
async def upload_markdown(request: Request, file: UploadFile = File(...)):
    """Uploads a structured Markdown file into the server's knowledge folder."""
    if not file.filename.endswith(".md"):
        raise HTTPException(status_code=400, detail="Only .md files are accepted.")
    
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_path = UPLOAD_DIR / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Admin uploaded knowledge file: '{file.filename}'")
        return {"status": "success", "message": f"Successfully uploaded {file.filename}"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def run_ingestion_script():
    logger.info("Starting background ingestion process...")
    try:
        # Run pipeline.ingest with -u (unbuffered output)
        process = subprocess.Popen(
            ["python", "-u", "-m", "pipeline.ingest"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout so errors stream in real-time too
            text=True,
            bufsize=1  # Line-buffered
        )

        # Read line by line as the script prints output
        for line in iter(process.stdout.readline, ""):
            cleaned_line = line.strip()
            if cleaned_line:
                # Logging automatically appends to RingBufferHandler and LOG_BUFFER
                logger.info(f"[Ingest] {cleaned_line}")

        process.stdout.close()
        return_code = process.wait()

        if return_code == 0:
            logger.info("Ingestion process completed successfully!")
        else:
            logger.error(f"Ingestion process failed with exit code: {return_code}")

    except Exception as e:
        logger.error(f"Failed to execute ingestion subprocess: {e}")
                
@app.post("/api/admin/ingest")
@limiter.limit("5/minute")
async def trigger_ingestion(request: Request, background_tasks: BackgroundTasks):
    """Triggers the ChromaDB vector ingestion process in a background thread."""
    background_tasks.add_task(run_ingestion_script)
    logger.info("Admin triggered database re-ingestion.")
    return {"status": "processing", "message": "Ingestion triggered in the background."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)