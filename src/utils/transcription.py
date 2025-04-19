import os
import whisper

def transcribe_video(video_path):
    # Optional: Set FFmpeg path if needed (or remove if managed via environment)
    ffmpeg_path = os.getenv("FFMPEG_PATH")
    if ffmpeg_path:
        os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

    model = whisper.load_model("tiny")
    print("🔍 Transcribing...")
    result = model.transcribe(video_path)
    return result["segments"]