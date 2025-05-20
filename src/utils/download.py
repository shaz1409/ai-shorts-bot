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
        # Step 1: Check availability
        ydl_check_opts = {
            'quiet': True,
            'skip_download': True,
        }
        with yt_dlp.YoutubeDL(ydl_check_opts) as ydl:
            info = ydl.extract_info(url, download=False)

        # Step 2: Filter only public videos
        availability = info.get("availability", "unknown")
        is_private = info.get("is_private", False)
        if is_private or availability not in ["public", None]:
            log.warning(f"🔒 Video rejected (availability: {availability}, private: {is_private})")
            raise ValueError("This video is not public or is unavailable.")

        # Step 3: Clean title
        raw_title = info.get('title', 'video')
        safe_title = sanitize_title(raw_title)
        log.info(f"🎬 Title: {raw_title}")
        log.info(f"🧼 Safe title: {safe_title}")

        # Step 4: Define output template
        output_template = os.path.join(output_dir, f"{safe_title}_{date_tag}.%(ext)s")

        # Step 5: Prepare yt-dlp with FFmpeg
        ffmpeg_path = os.getenv("FFMPEG_PATH") or "ffmpeg"
        log.info(f"🛠 Using ffmpeg from: {ffmpeg_path}")

        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
            'merge_output_format': 'mp4',
            'ffmpeg_location': ffmpeg_path,
            'outtmpl': output_template,
            'quiet': False,
            'cookiefile': 'cookies.txt'  # 👈 This is key!
        }

        if not os.path.exists("cookies.txt"):
            log.warning("⚠️ cookies.txt not found in expected path!")
        else:
            log.info("🍪 cookies.txt found and will be used.")



        # Step 6: Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Step 7: Return downloaded file
        for file in os.listdir(output_dir):
            if file.endswith(".mp4") and file.startswith(safe_title):
                log.info(f"📁 Video saved: {file}")
                return os.path.join(output_dir, file), os.path.splitext(file)[0]

        log.warning("⚠️ No MP4 file found after download.")
        return None, None

    except Exception as e:
        log.error(f"❌ Failed to download video: {e}")
        return None, None
