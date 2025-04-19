import os
import json
from datetime import datetime
from dotenv import load_dotenv
from src.utils.download import download_youtube_video
from src.utils.transcription import transcribe_video
from src.gpt_matching import find_clip_segments
from src.video_processing import cut_clips

load_dotenv()

def generate_shorts(youtube_url):
    video_path, title_safe = download_youtube_video(youtube_url)
    print("✅ Video downloaded:", video_path)

    segments = transcribe_video(video_path)
    highlights = find_clip_segments(segments)
    clips = cut_clips(video_path, highlights, title_safe, segments)

    date_tag = datetime.now().strftime("%d%m%Y")
    metadata_path = f"output/shorts_metadata_{title_safe}_{date_tag}.json"
    with open(metadata_path, "w") as f:
        json.dump(clips, f, indent=2)

    print(f"✅ {len(highlights)} highlight clip(s) created from GPT quotes.")
    if not highlights:
        print("❌ No valid highlights found. Try a different video.")
    if highlights:
        print("🎉 Shorts created and saved to 'output/'")
    return clips

if __name__ == "__main__":
    url = input("📥 Enter YouTube video URL: ").strip()
    generate_shorts(url)
