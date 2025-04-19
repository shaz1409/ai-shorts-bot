import os
import re
import yt_dlp
from datetime import datetime
from src.logging_config import log

def sanitize_title(title):
    return re.sub(r'[^a-zA-Z0-9\-]+', '-', title.strip().lower())[:60]

def download_youtube_video(url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    date_tag = datetime.now().strftime("%d%m%Y")

    log.info(f"🔗 Attempting to download video: {url}")

    try:
        ydl_check_opts = {
            'quiet': True,
            'skip_download': True,
            'extract_flat': True,
        }
        with yt_dlp.YoutubeDL(ydl_check_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("is_private") or info.get("availability") == "private":
                raise ValueError("Video is private or unavailable")
            raw_title = info.get('title', 'video')
            safe_title = sanitize_title(raw_title)
            log.info(f"🎬 Raw title: {raw_title}")
            log.info(f"🧼 Safe title: {safe_title}")

        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get('availability') not in ['public', None]:
                raise ValueError("Video is not public or available.")
                log.warning(f"🔒 Video not public: {info.get('availability')}")


        output_template = os.path.join(output_dir, f"{safe_title}_{date_tag}.%(ext)s")

        ffmpeg_path = os.getenv("FFMPEG_PATH") or "ffmpeg"
        log.info(f"🛠 Using ffmpeg from: {ffmpeg_path}")

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
                log.info(f"📁 Found video: {file}")
                return os.path.join(output_dir, file), os.path.splitext(file)[0]

        log.warning("⚠️ No MP4 file found after download.")
        return None, None

    except Exception as e:
        log.error(f"❌ Failed to download video: {e}")
        return None, None
