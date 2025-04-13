# src/captioning.py

import tempfile
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def get_captions_for_clip(start, end, segments):
    return [seg for seg in segments if seg['start'] >= start and seg['end'] <= end]

def style_caption(text, start, end, video_width, video_height=1080):
    try:
        font_size = 72
        font_path = "arial.ttf"
        chunk_size = 4
        delay = 1.0

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
