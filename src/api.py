from fastapi import FastAPI, BackgroundTasks, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uuid import uuid4
import os
import zipfile
import io

from src.services.video_generator import generate_video_pipeline
from src.logging_config import log

app = FastAPI()
templates = Jinja2Templates(directory="templates")

OUTPUT_DIR = "output"
JOBS = {}

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    log.info("📄 Serving form page")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-form")
async def handle_form(
    request: Request,
    url: str = Form(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    job_id = str(uuid4())
    job_path = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    JOBS[job_id] = {
        "status": "processing",
        "error": None,
        "output": []
    }

    log.info(f"🚀 New form job submitted: {job_id} | URL: {url}")
    background_tasks.add_task(generate_video_pipeline, url, job_path, job_id, JOBS)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "job_id": job_id,
        "status": "processing"
    })

class GenerateRequest(BaseModel):
    url: str

@app.post("/generate")
async def generate(req: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    job_path = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_path, exist_ok=True)

    JOBS[job_id] = {
        "status": "processing",
        "error": None,
        "output": []
    }

    log.info(f"🚀 New API job submitted: {job_id} | URL: {req.url}")
    background_tasks.add_task(generate_video_pipeline, req.url, job_path, job_id, JOBS)

    return {"job_id": job_id, "status": "processing"}

@app.post("/download-selected")
async def download_selected(request: Request):
    form_data = await request.form()
    selected_files = form_data.getlist("selected_videos")

    if not selected_files:
        return {"error": "No videos selected."}

    log.info(f"📦 Zipping {len(selected_files)} selected videos...")

    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, "w") as zipf:
        for file_path in selected_files:
            filename = os.path.basename(file_path)
            full_path = os.path.join("output", file_path)
            if os.path.exists(full_path):
                zipf.write(full_path, arcname=filename)
            else:
                log.warning(f"⚠️ File not found: {full_path}")

    memory_file.seek(0)

    return StreamingResponse(
        memory_file,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment; filename=shorts.zip"}
    )

@app.get("/status/{job_id}")
async def status(job_id: str):
    job = JOBS.get(job_id)

    if not isinstance(job, dict):
        log.warning(f"❓ Malformed or missing job entry for: {job_id}")
        return {"job_id": job_id, "status": "not found"}

    log.debug(f"📦 Status check — Job {job_id}: {job.get('status')}")

    return {
        "job_id": job_id,
        "status": job.get("status"),
        "error": job.get("error"),
        "output": job.get("output", [])
    }

@app.get("/video/{job_id}")
async def get_video(job_id: str):
    path = os.path.join(OUTPUT_DIR, job_id, "final.mp4")
    if os.path.exists(path):
        log.info(f"📥 Serving downloadable video for job {job_id}")
        return FileResponse(
            path,
            media_type="video/mp4",
            filename=f"{job_id}_short.mp4"
        )
    else:
        log.warning(f"❌ Video not found for job {job_id}")
        return {"error": "Video not ready or job not found"}

@app.post("/zip/{job_id}")
async def zip_selected_videos(job_id: str, request: Request):
    data = await request.json()
    selected_files = data.get("files", [])

    if not selected_files:
        return JSONResponse(content={"error": "No files selected"}, status_code=400)

    job_path = os.path.join(OUTPUT_DIR, job_id)
    zip_path = os.path.join(job_path, f"{job_id}_selected_clips.zip")

    try:
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for fname in selected_files:
                full_path = os.path.join(job_path, fname)
                if os.path.exists(full_path):
                    zipf.write(full_path, arcname=fname)
        log.info(f"🗜️ Created zip: {zip_path}")
        return {"download": f"/static/{job_id}/{job_id}_selected_clips.zip"}
    except Exception as e:
        log.error(f"❌ Failed to create zip for job {job_id}: {e}")
        return JSONResponse(content={"error": "Zip creation failed"}, status_code=500)

app.mount("/static", StaticFiles(directory="output"), name="static")
