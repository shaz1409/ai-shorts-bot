# 🎬 AI Shorts Bot

Automatically generate viral, mobile-friendly YouTube Shorts or TikToks from long-form YouTube videos.  
Powered by GPT-3.5, Whisper, OpenCV, and MoviePy — fully automated and smart.

---

## 🚀 Features

- ✅ Downloads full YouTube videos
- 🧠 Transcribes with OpenAI Whisper
- ✂️ Extracts highlight-worthy quotes using GPT-3.5
- 📊 Sentiment-based fallback if GPT fails
- 🎯 Match quotes to transcript intelligently
- 📼 Auto-cuts and edits clips
- 💬 Stylized captions with chunking + dynamic coloring
- 🧠 Tracks faces and centers on speakers
- 📱 Exports vertical 9:16 videos for YouTube Shorts & TikTok

---

## 📦 Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/YOUR-USERNAME/ai-shorts-bot.git
   cd ai-shorts-bot
   ```

2. **Create your `.env` file**
   ```bash
   cp .env.example .env
   ```

   Update it with your OpenAI key and FFmpeg path:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   FFMPEG_PATH=C:/path/to/ffmpeg/bin
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download FFmpeg**
   - [FFmpeg Windows Builds](https://www.gyan.dev/ffmpeg/builds/)
   - Extract and point to the `bin` folder in your `.env`

---

## ▶️ How to Use

Run the bot:
```bash
python src/main.py
```

Paste a YouTube video URL when prompted. The bot will:
- Download the video
- Transcribe with Whisper
- Use GPT-3.5 to detect highlight moments
- Automatically cut, crop, and style each clip
- Add dynamic captions
- Export clips to `output/`

---

## 📁 Folder Structure

```
ai-shorts-bot/
├── src/
│   ├── main.py
│   ├── gpt_matching.py
│   ├── video_processing.py
│   └── utils/
│       ├── download.py
│       ├── transcription.py
│       └── __init__.py
├── .env.example
├── requirements.txt
├── README.md
```

---

## 💡 Example Output

Output clips saved in:
```
output/
├── your-video_short_14042025_1.mp4
├── your-video_short_14042025_2.mp4
```

Each short is:
- 45–60 seconds long  
- Formatted for mobile (1080x1920)  
- Includes punchy, styled captions

---

## ⚠️ Notes

- Requires OpenAI API key (free or paid account)
- Whisper model runs on CPU/GPU depending on your system
- Face tracking uses OpenCV for smart cropping
- Code is modular and production-ready

---

## 🛠️ To Do

- [ ] Add support for custom clip prompts (e.g. "focus on goals" for football)
- [ ] Web UI for upload + download
- [ ] Auto-post to Instagram / TikTok

---

## 🧠 Credits

Built with ❤️ by [@shaz1409](https://github.com/shaz1409)

---
