def generate_video_pipeline(url, output_path, job_id, job_store):
    try:
        from src.utils.download import download_youtube
        from src.utils.transcription import transcribe_audio
        from src.gpt_matching import extract_highlights
        from src.captioning import generate_captions
        from src.video_processing import export_final_video

        # You can refine this pipeline later
        audio_path, video_path = download_youtube(url, output_path)
        transcript = transcribe_audio(audio_path)
        highlights = extract_highlights(transcript)
        captioned_segments = generate_captions(highlights)
        export_final_video(video_path, captioned_segments, output_path)

        job_store[job_id] = "complete"
    except Exception as e:
        job_store[job_id] = f"error: {str(e)}"
