# src/gpt_matching.py

import os
import re
import json
import openai
import whisper
from difflib import SequenceMatcher
from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")

def transcribe_video(video_path):
    ffmpeg_dir = os.getenv("FFMPEG_PATH")
    if ffmpeg_dir:
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ["PATH"]
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

        parsed = json.loads(output)
        parsed.sort(key=lambda q: q.get('rating', 0), reverse=True)

        best_quote = parsed[0:1] if parsed else []
        strong_quotes = [q for q in parsed[1:] if q.get("rating", 0) >= 7]

        print("🧠 Best quote:", best_quote)
        print("📊 Strong quotes (7+):", len(strong_quotes))

        return best_quote + strong_quotes[:4]

    except Exception as e:
        print("⚠️ GPT fallback error:", str(e))
        return []

def expand_segment_to_clip(start_index, segments, min_len=45, max_len=60):
    start_time = segments[start_index]['start']
    end_time = start_time
    i = start_index

    while i < len(segments) and (end_time - start_time) < max_len:
        end_time = segments[i]['end']
        if (end_time - start_time) >= min_len and segments[i]['text'].strip().endswith(('.', '?', '!')):
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
        return " ".join(seg['text'] for seg in segments[start_idx:start_idx+window]).strip()

    for quote_obj in quotes:
        quote = quote_obj["quote"]
        found = False

        for idx in range(len(segments)):
            combined_text = combine_segments(idx)

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