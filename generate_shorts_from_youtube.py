import os
import json
import re
import openai
import yt_dlp
import whisper
from datetime import datetime
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from transformers import pipeline
from dotenv import load_dotenv
from difflib import SequenceMatcher
from PIL import Image, ImageDraw, ImageFont
import textwrap
import tempfile
import cv2
import numpy as np
from scipy.interpolate import interp1d

import moviepy.config as mpy_config
mpy_config.change_settings({"IMAGEMAGICK_BINARY": None})

# Load OpenAI API Key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Sentiment pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

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

    ffmpeg_path = r"C:\Users\sahmed\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best',
        'merge_output_format': 'mp4',
        'ffmpeg_location': ffmpeg_path,
        'outtmpl': os.path.join(output_dir, f"{safe_title}_{date_tag}.%(ext)s"),
        'quiet': False  # ✅ show full logs for debug
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    for file in os.listdir(output_dir):
        if file.endswith(".mp4"):
            print("📁 Found video:", file)
            return os.path.join(output_dir, file), os.path.splitext(file)[0]

    
    return None, None

def transcribe_video(video_path):
    ffmpeg_path = r'C:\Users\sahmed\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin'
    os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]

    model = whisper.load_model("base")
    print("🔍 Transcribing...")
    result = model.transcribe(video_path)
    return result["segments"]

def gpt_highlight_quotes(full_text):
    try:
        client = openai.OpenAI()

        messages = [
            {
                "role": "system",
                "content": (
                    "You're an expert short-form content editor for YouTube and TikTok. "
                    "Your job is to select up to 5 strong moments from a video transcript that would make viral YouTube Shorts. "
                    "These should be emotionally charged, surprising, insightful, or highly relatable moments — NOT filler, intros, or common statements. "
                    "For each quote, also assign a virality rating from 1 to 10."
                )
            },
            {
                "role": "user",
                "content": (
                    "From the transcript below, extract up to 5 highlight-worthy quotes. "
                    "Each quote should be a standalone sentence (or two), less than 60 words, and include a 'rating' field (1–10). "
                    "Only return the result as a JSON list, like this:\n\n"
                    '[{"quote": "Your moment here.", "rating": 9}, ...]\n\nTranscript:\n' + full_text
                )
            }
        ]

        print("🧐 GPT-3.5 assisting with highlight detection...")
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.3
        )

        output = response.choices[0].message.content.strip()
        output = output.replace("```json", "").replace("```", "").strip()
        if output.lower().startswith("json"):
            output = output[4:].strip()

        try:
            parsed = json.loads(output)

            # Sort by rating (desc)
            parsed.sort(key=lambda q: q.get('rating', 0), reverse=True)

            # Always include the top quote
            best_quote = parsed[0:1] if parsed else []

            # Add others only if rated 7+
            strong_quotes = [q for q in parsed[1:] if q.get("rating", 0) >= 7]

            print("🧠 Best quote:", best_quote)
            print("📊 Strong quotes (7+):", len(strong_quotes))

            return best_quote + strong_quotes[:4]

        except json.JSONDecodeError:
            print("⚠️ GPT responded with invalid JSON:\n", output)
            return []

    except Exception as e:
        print("⚠️ GPT fallback error:", str(e))
        return []

def expand_segment_to_clip(start_index, segments, min_len=45, max_len=60):
    start_time = segments[start_index]['start']
    end_time = start_time
    i = start_index

    while i < len(segments) and (end_time - start_time) < max_len:
        end_time = segments[i]['end']
        duration = end_time - start_time

        if duration >= min_len and segments[i]['text'].strip().endswith(('.', '?', '!')):
            break
        i += 1

    return (start_time, end_time)

def is_similar(a, b, threshold=0.85):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold

def loose_keyword_match(quote, segment_text):
    quote_keywords = set(re.sub(r"[^\w\s]", "", quote).lower().split())
    segment_keywords = set(re.sub(r"[^\w\s]", "", segment_text).lower().split())
    if not quote_keywords:
        return False
    overlap = quote_keywords.intersection(segment_keywords)
    return len(overlap) / len(quote_keywords) >= 0.4

def match_quotes_to_segments(quotes, segments):
    print("🔗 Matching GPT quotes to transcript...")
    matched_clips = []

    def combine_segments(start_idx, window=3):
        combined = " ".join(seg['text'] for seg in segments[start_idx:start_idx+window])
        return combined.strip()

    for quote_obj in quotes:
        quote = quote_obj["quote"]
        found = False

        for idx in range(len(segments)):
            combined_text = combine_segments(idx, window=3)

            print("🔍 Trying to match:")
            print("   📢 GPT Quote:", quote)
            print("   📜 Transcript Chunk:", combined_text)

            if is_similar(quote, combined_text) or loose_keyword_match(quote, combined_text):
                clip = expand_segment_to_clip(idx, segments)
                matched_clips.append(clip)

                print(f"\n🎯 Matched quote: \"{quote}\" (Rating: {quote_obj['rating']})")
                print(f"🎬 Cutting from {clip[0]:.1f}s to {clip[1]:.1f}s (length: {clip[1] - clip[0]:.1f}s)")
                found = True
                break

        if not found:
            print(f"❌ Could not match quote: {quote}")

        if len(matched_clips) == 5:
            break

    return matched_clips

def fallback_sentiment_selection(segments):
    print("🤖 Using fallback: sentiment-based selection...")
    results = []

    for seg in segments:
        text = seg['text']
        if len(text.strip()) < 10:
            continue

        sentiment = sentiment_pipeline(text[:512])[0]
        label = sentiment['label']
        score = sentiment['score']

        weight = score if label == "POSITIVE" else score * 0.5
        results.append((weight, seg['start'], seg['start'] + 60, text))

    results.sort(reverse=True)
    return [(start, end) for _, start, end, _ in results[:5]]

def find_clip_segments(segments):
    full_text = " ".join([seg['text'] for seg in segments])
    quotes = gpt_highlight_quotes(full_text)

    if quotes:
        return match_quotes_to_segments(quotes, segments)
    else:
        return fallback_sentiment_selection(segments)

def get_captions_for_clip(start, end, segments):
    return [seg for seg in segments if seg['start'] >= start and seg['end'] <= end]

def style_caption(text, start, end, video_width, video_height=1080):
    try:
        from transformers import pipeline
        font_size = 72
        font_path = "arial.ttf"
        chunk_size = 4
        delay = 1.0

        sentiment_pipeline = pipeline("sentiment-analysis")

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
            print("⚠️ Using default font (Arial not found)")

        words = text.strip().split()
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        clips = []
        y_pos = video_height - 320

        for i, phrase in enumerate(chunks):
            # Analyze sentiment for this phrase
            try:
                result = sentiment_pipeline(phrase[:512])[0]
                label = result['label']
            except:
                label = "NEUTRAL"

            if label == "POSITIVE":
                fill = "lime"
            elif label == "NEGATIVE":
                fill = "red"
            else:
                fill = "white"

            img_width = int(video_width * 0.9)
            img_height = font_size + 60
            img = Image.new("RGBA", (img_width, img_height), (0, 0, 0, 180))
            draw = ImageDraw.Draw(img)
            bbox = draw.textbbox((0, 0), phrase, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            draw.text(((img.width - w) / 2 + 2, 20 + 2), phrase, font=font, fill="black")
            draw.text(((img.width - w) / 2, 20), phrase, font=font, fill=fill)

            temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(temp_file.name)

            chunk_clip = (
                ImageClip(temp_file.name)
                .set_start(start + i * delay)
                .set_duration(1.8)
                .set_position(("center", y_pos))
                .crossfadein(0.1)
            )
            clips.append(chunk_clip)

        return clips

    except Exception as e:
        print(f"❌ Styled caption render failed for: {text} | Error: {e}")
        return []

def classify_video_scene(subclip, sample_frames=5):
    duration = subclip.duration
    timestamps = np.linspace(0, duration, sample_frames)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_counts = []
    for t in timestamps:
        try:
            frame = subclip.get_frame(t)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            face_counts.append(len(faces))
        except:
            face_counts.append(0)

    avg_faces = np.mean(face_counts)
    print(f"👥 Avg faces per frame: {avg_faces:.2f}")
    return avg_faces

def track_face_x_centers(subclip, num_samples=15):
    duration = subclip.duration
    timestamps = np.linspace(0, duration, num_samples)
    centers = []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for t in timestamps:
        try:
            frame = subclip.get_frame(t)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                centers.append(x + w // 2)
            else:
                centers.append(None)
        except:
            centers.append(None)

    smoothed = []
    for i, c in enumerate(centers):
        if c is not None:
            smoothed.append(c)
        else:
            prev = next((centers[j] for j in range(i - 1, -1, -1) if centers[j] is not None), None)
            next_val = next((centers[j] for j in range(i + 1, len(centers)) if centers[j] is not None), None)
            if prev and next_val:
                smoothed.append(int((prev + next_val) / 2))
            elif prev:
                smoothed.append(prev)
            elif next_val:
                smoothed.append(next_val)
            else:
                smoothed.append(None)

    return timestamps, smoothed

def detect_speaking_face_x(subclip, segment_start, segment_end):
    try:
        mid_time = (segment_start + segment_end) / 2
        frame = subclip.get_frame(mid_time)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        return x + w // 2
    except Exception as e:
        print(f"❌ Error in speaking face detection: {e}")
        return None

def dynamic_face_crop(subclip, target_w, target_h):
    print("🧠 Running dynamic face tracking...")

    timestamps, centers = track_face_x_centers(subclip)
    if not centers or all(c is None for c in centers):
        print("🛑 No valid face centers detected, falling back to center crop.")
        return subclip.crop(x_center=subclip.w // 2, width=target_w, height=target_h)

    # Interpolate centers into a smooth function over time
    times = np.array([t for t, c in zip(timestamps, centers) if c is not None])
    values = np.array([c for c in centers if c is not None])
    interp_func = np.interp

    def face_x_func(t):
        return float(interp_func(t, times, values))

    return subclip.crop(
        x_center=lambda t: face_x_func(t),
        width=target_w,
        height=target_h
    )

def cut_clips(video_path, segments, title_safe, transcript_segments, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    full_clip = VideoFileClip(video_path)
    generated = []
    date_tag = datetime.now().strftime("%d%m%Y")

    for i, (start, end) in enumerate(segments):
        if (end - start) < 8:
            print(f"⚠️ Clip too short ({end - start:.1f}s), skipping.")
            continue

        short_path = os.path.join(output_dir, f"{title_safe}_short_{date_tag}_{i+1}.mp4")
        subclip = full_clip.subclip(start, end)

        # Resize for mobile vertical
        target_w, target_h = 1080, 1920
        subclip = subclip.resize(height=target_h)

        print("📍 Applying dynamic face crop...")

        # Track face centers over time
        times, centers = track_face_x_centers(subclip, num_samples=20)
        if any(c is not None for c in centers):
            print("📈 Running dynamic face tracking...")
            valid_times = [t for t, x in zip(times, centers) if x is not None]
            valid_centers = [x for x in centers if x is not None]

            if len(valid_times) > 1:
                face_x_func = interp1d(valid_times, valid_centers, kind="linear", fill_value="extrapolate")
                subclip = subclip.crop(x_center=lambda t: float(face_x_func(t)), width=target_w, height=target_h)
            else:
                print("📍 Not enough valid face points — fallback to center crop.")
                subclip = subclip.crop(x_center=subclip.w // 2, width=target_w, height=target_h)
        else:
            print("❌ No face detected — fallback to center crop.")
            subclip = subclip.crop(x_center=subclip.w // 2, width=target_w, height=target_h)

        # Captions
        captions = get_captions_for_clip(start, end, transcript_segments)
        caption_clips = []
        for seg in captions:
            styled = style_caption(seg['text'], seg['start'] - start, seg['end'] - start, target_w, target_h)
            if styled:
                caption_clips.extend(styled)

        if not caption_clips:
            print("⚠️ Skipping clip — all captions failed to render.")
            continue

        print("📦 Final composite layers:")
        for clip in [subclip] + caption_clips:
            print(" -", type(clip), getattr(clip, 'duration', None))

        final = CompositeVideoClip([subclip] + caption_clips, size=(target_w, target_h))
        final = final.set_audio(subclip.audio)
        final.write_videofile(short_path, fps=24, codec="libx264", audio_codec="aac")

        print(f"✅ Exported: {short_path}")

        generated.append({
            "file_path": short_path,
            "start": start,
            "end": end
        })

    return generated

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
