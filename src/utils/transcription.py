import whisper
import os
from src.logging_config import log

def transcribe_video(video_path):
    log.info(f"📼 Starting transcription for: {video_path}")

    if not os.path.exists(video_path):
        log.error(f"🚫 Video file does not exist: {video_path}")
        raise FileNotFoundError(f"Video not found: {video_path}")

    try:
        model = whisper.load_model("base")
        log.info("🧠 Whisper model loaded.")
        result = model.transcribe(video_path)
        log.info(f"✅ Transcription completed. Tokens: {len(result['segments'])}")

        return result["segments"]
    except Exception as e:
        log.exception(f"❌ Error during transcription: {e}")
        raise
