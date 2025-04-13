from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

from src.utils.download import download_youtube_video
from src.utils.transcription import transcribe_video
from src.gpt_matching import find_clip_segments
from src.video_processing import cut_clips

app = FastAPI()

# Template location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "..", "templates"))

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate", response_class=HTMLResponse)
async def generate_video(request: Request, url: str = Form(...)):
    video_path, title_safe = download_youtube_video(url)
    segments = transcribe_video(video_path)
    highlights = find_clip_segments(segments)
    clips = cut_clips(video_path, highlights, title_safe, segments)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "message": f"✅ {len(clips)} short(s) generated for: {title_safe}"
    })
