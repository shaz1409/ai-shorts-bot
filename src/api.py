from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
from uuid import uuid4
import os
from src.services.video_generator import generate_video_pipeline

app = FastAPI()

OUTPUT_DIR = "output"
JOBS = {}  # Basic in-memory status store for now

class GenerateRequest(BaseModel):
    url: str

@app.post("/generate")
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    job_path = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    JOBS[job_id] = "processing"
    background_tasks.add_task(generate_video_pipeline, req.url, job_path, job_id, JOBS)

    return {"job_id": job_id, "status": "processing"}

@app.get("/status/{job_id}")
async def status(job_id: str):
    status = JOBS.get(job_id, "not found")
    return {"job_id": job_id, "status": status}

@app.get("/video/{job_id}")
async def get_video(job_id: str):
    path = os.path.join(OUTPUT_DIR, job_id, "final.mp4")
    if os.path.exists(path):
        return {"download": f"/static/{job_id}/final.mp4"}
    else:
        return {"error": "Video not ready or job not found"}
