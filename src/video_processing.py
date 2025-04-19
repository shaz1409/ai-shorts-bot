# src/video_processing.py

import os
import cv2
import numpy as np
from datetime import datetime
from scipy.interpolate import interp1d
from moviepy.editor import VideoFileClip, CompositeVideoClip
from src.captioning import get_captions_for_clip, style_caption
from src.logging_config import log
import copy

def track_face_x_centers(subclip, num_samples=20):
    log.info("Tracking Face...")
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
        except Exception as e:
            log.error(f"Face tracking failed at timestamp {t}: {e}")
            centers.append(None)
    log.info("Found the Faces!!!")
    return timestamps, centers

def dynamic_face_crop(subclip, target_w, target_h):
    log.info("📍 Starting dynamic face crop...")
    times, centers = track_face_x_centers(subclip)

    log.info(f"🕵️ Total timestamps: {len(times)}, Total centers: {len(centers)}")
    log.info(f"📊 Raw centers: {centers}")

    if any(c is not None for c in centers):
        valid_times = [t for t, x in zip(times, centers) if x is not None]
        valid_centers = [x for x in centers if x is not None]

        log.info(f"✅ Valid face detections: {len(valid_centers)}")
        log.info(f"⏱️ Valid times: {valid_times}")
        log.info(f"🎯 Valid centers: {valid_centers}")

        if len(valid_times) > 1:
            log.info(f"📈 Interpolating {len(valid_centers)} face positions...")
            face_x_func = interp1d(valid_times, valid_centers, kind="linear", fill_value="extrapolate")

            def safe_crop_t(t):
                log.debug(f"🌀 Lambda called — t={t} ({type(t)})")
                try:
                    result = float(face_x_func(t))
                    log.debug(f"✅ Interpolated x_center: {result}")
                    return result
                except Exception as e:
                    log.error(f"❌ Lambda interpolation failed at t={t}: {e}")
                    raise

            try:
                cropped = subclip.crop(x_center=safe_crop_t, width=target_w, height=target_h)
                log.info("🎬 Returning face-tracked crop ✅")
                return cropped
            except Exception as e:
                log.warning(f"⚠️ Error during face-tracked crop: {e}")
                log.info("↪️ Falling back to center crop.")
        else:
            log.warning("📍 Not enough valid face points — fallback to center crop.")
    else:
        log.warning("❌ No faces detected — fallback to center crop.")

    fallback_x = subclip.w // 2
    log.info(f"🎯 Fallback center x = {fallback_x}")
    cropped = subclip.crop(x_center=fallback_x, width=target_w, height=target_h)
    log.info("🎬 Returning fallback center crop ✅")
    return cropped

def cut_clips(video_path, segments, title_safe, transcript_segments, output_dir="output"):
    log.info("🔁 Reloaded!")
    os.makedirs(output_dir, exist_ok=True)
    full_clip = VideoFileClip(video_path)
    generated = []
    date_tag = datetime.now().strftime("%d%m%Y")

    target_w, target_h = 1080, 1920

    for i, (clip_start, clip_end) in enumerate(segments):
        log.info(f"🧮 clip_start = {clip_start} ({type(clip_start)})")

        if (clip_end - clip_start) < 8:
            log.warning(f"⚠️ Clip too short ({clip_end - clip_start:.1f}s), skipping.")
            continue

        filename = f"{title_safe}_short_{date_tag}_{i+1}.mp4"
        if filename.startswith("-"):
            filename = "_" + filename

        short_path = os.path.join(output_dir, filename)

        subclip = full_clip.subclip(clip_start, clip_end)
        subclip = subclip.resize(height=target_h)
        subclip = dynamic_face_crop(subclip, target_w, target_h)
        log.info("Dynamic Face Crop Worked...")

        captions = get_captions_for_clip(clip_start, clip_end, copy.deepcopy(transcript_segments))
        log.info(f"🧾 Got {len(captions)} captions from get_captions_for_clip")
        log.info(f"🧾 First caption example: {captions[0] if captions else 'None'}")
        log.debug(f"📚 All captions: {captions}")

        caption_clips = []

        for seg in captions:
            try:
                log.debug(f"🧩 Raw caption: {seg} (type={type(seg)})")

                if not isinstance(seg, dict):
                    log.warning("🚨 Skipping: seg is not a dict!")
                    continue

                if not isinstance(seg.get('start'), (int, float)) or not isinstance(seg.get('end'), (int, float)) or not isinstance(clip_start, (int, float)):
                    log.warning(f"⚠️ Skipping caption with invalid timing types: seg={seg}, clip_start={clip_start}")
                    continue

                log.debug(f"🧪 Types ⇒ seg['start']: {type(seg['start'])}, seg['end']: {type(seg['end'])}, clip_start: {type(clip_start)}")

                start_offset = seg['start'] - clip_start
                end_offset = seg['end'] - clip_start

                styled = style_caption(seg['text'], start_offset, end_offset, target_w, target_h)
                if styled:
                    caption_clips.extend(styled)

            except Exception as e:
                log.error(f"⚠️ Skipping broken caption: {seg} | Error: {e}")

        if not caption_clips:
            log.warning("⚠️ Skipping clip — all captions failed to render.")
            continue

        try:
            final = CompositeVideoClip([subclip] + caption_clips, size=(target_w, target_h))
            final = final.set_audio(subclip.audio)
            final.write_videofile(short_path, fps=24, codec="libx264", audio_codec="aac")
            log.info(f"✅ Exported: {short_path}")

            generated.append({
                "file_path": short_path,
                "start": clip_start,
                "end": clip_end
            })
        except Exception as e:
            log.error(f"❌ Failed to export final clip: {e}")

    log.info(f"🧵 Done — {len(generated)} clip(s) exported.")
    return generated