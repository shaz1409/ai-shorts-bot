# src/video_processing.py

import os
import cv2
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from moviepy.editor import VideoFileClip, CompositeVideoClip
from captioning import get_captions_for_clip, style_caption


def track_face_x_centers(subclip, num_samples=20):
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

    return timestamps, centers

def dynamic_face_crop(subclip, target_w, target_h):
    print("📍 Applying dynamic face crop...")
    times, centers = track_face_x_centers(subclip)
    if any(c is not None for c in centers):
        valid_times = [t for t, x in zip(times, centers) if x is not None]
        valid_centers = [x for x in centers if x is not None]

        if len(valid_times) > 1:
            face_x_func = interp1d(valid_times, valid_centers, kind="linear", fill_value="extrapolate")
            return subclip.crop(x_center=lambda t: float(face_x_func(t)), width=target_w, height=target_h)
        else:
            print("📍 Not enough face points — fallback to center crop.")
    else:
        print("❌ No faces detected — fallback to center crop.")

    return subclip.crop(x_center=subclip.w // 2, width=target_w, height=target_h)

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

        # Resize for vertical
        target_w, target_h = 1080, 1920
        subclip = subclip.resize(height=target_h)
        subclip = dynamic_face_crop(subclip, target_w, target_h)

        captions = get_captions_for_clip(start, end, transcript_segments)
        caption_clips = []
        for seg in captions:
            styled = style_caption(seg['text'], seg['start'] - start, seg['end'] - start, target_w, target_h)
            if styled:
                caption_clips.extend(styled)

        if not caption_clips:
            print("⚠️ Skipping clip — all captions failed to render.")
            continue

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