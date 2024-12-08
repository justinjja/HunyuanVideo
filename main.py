import os
import time
import uuid
from datetime import datetime
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import shlex

app = FastAPI()

# Ensure results directory exists
os.makedirs("./results", exist_ok=True)

app.mount("/results", StaticFiles(directory="./results"), name="results")

def validate_params(height: int, width: int, length: int):
    if not (1 <= height <= 720):
        raise HTTPException(status_code=400, detail="Height must be between 1 and 720.")
    if not (1 <= width <= 1280):
        raise HTTPException(status_code=400, detail="Width must be between 1 and 1280.")
    if not (41 <= length <= 121):
        raise HTTPException(status_code=400, detail="Length must be between 41 and 121.")

@app.get("/")
def read_index():
    return FileResponse("index.html")

@app.post("/generate")
async def generate_video(
    height: int = Form(...),
    width: int = Form(...),
    length: int = Form(...),
    prompt: str = Form(...)
):
    validate_params(height, width, length)

    # Construct a unique run identifier
    run_id = datetime.now().strftime("%Y-%m-%d-%H:%M:%S") + f"_{uuid.uuid4().hex[:6]}"
    # We'll let sample_video.py generate the final filename, but we can capture that from logs.

    cmd = f"stdbuf -oL python3 sample_video.py --video-size {height} {width} --video-length {length} --infer-steps 50 --prompt \"{prompt}\" --flow-reverse --use-cpu-offload --save-path ./results"
    # Using asyncio subprocess for streaming output
    process = await asyncio.create_subprocess_exec(
        *shlex.split(cmd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )

    async def stream_stdout() -> AsyncGenerator[str, None]:
        # Stream output line by line
        if process.stdout:
            async for line in process.stdout:
                line_text = line.decode('utf-8', errors='replace')
                yield line_text
        await process.wait()

    return StreamingResponse(stream_stdout(), media_type="text/plain")
