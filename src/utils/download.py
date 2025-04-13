import os
import re
import yt_dlp
from datetime import datetime

def sanitize_title(title):
    return re.sub(r'[^a-zA-Z0-9\-]+', '-', title.strip().lower())[:60]

def download_youtube_video(url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    date_tag = datetime.now().strftime("%d%m%Y")

    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        raw_title = info.get('title', 'video')
        safe_title = sanitize_title(raw_title)

    output_template = os.path.join(output_dir, f"{safe_title}_{date_tag}.%(ext)s")

    ffmpeg_path = os.getenv("FFMPEG_PATH") or "ffmpeg"

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_path,
        'outtmpl': output_template,
        'quiet': False
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    for file in os.listdir(output_dir):
        if file.endswith(".mp4"):
            print("📁 Found video:", file)
            return os.path.join(output_dir, file), os.path.splitext(file)[0]

    return None, None
