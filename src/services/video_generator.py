# src/services/video_generator.py

import os
from datetime import datetime
from src.utils.download import download_youtube_video
from src.utils.transcription import transcribe_video
from src.gpt_matching import find_clip_segments
from src.video_processing import cut_clips
import shutil

from src.logging_config import log

def generate_video_pipeline(url, output_path, job_id, job_store):
    try:
        log.info(f"🚀 Starting job {job_id}")
        video_path, title_safe = download_youtube_video(url, output_dir=output_path)

        if not video_path:
            raise ValueError("Failed to download video")

        log.info(f"✅ Downloaded: {video_path}")
        segments = transcribe_video(video_path)
        highlights = find_clip_segments(segments)

        if not highlights:
            raise ValueError("No valid highlights found")

        log.info(f"🎯 Found {len(highlights)} highlight(s)")
        log.info("🎬 About to cut clips...")

        final_clips = cut_clips(video_path, highlights, title_safe, segments, output_dir=output_path)

        if not final_clips:
            raise ValueError("No clips were generated")

        # Copy the best (first) clip as final.mp4 for API return
        best_clip_path = final_clips[0]["file_path"]
        final_output = os.path.join(output_path, "final.mp4")
        shutil.copy(best_clip_path, final_output)

        log.info(f"🎉 Final video ready at {final_output}")

        # Update job store with full details
        job_store[job_id] = {
            "status": "complete",
            "error": None,
            "output": [os.path.basename(clip["file_path"]) for clip in final_clips]
        }

    except Exception as e:
        log.error(f"❌ Error during job {job_id}: {str(e)}")
        job_store[job_id] = {
            "status": f"error: {str(e)}",
            "error": str(e),
            "output": None
        }
