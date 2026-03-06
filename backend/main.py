import uuid
import json
import subprocess
import shutil
import threading
import re
import struct
import wave
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import yt_dlp

app = FastAPI(title="ClipForge API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Directories ───────────────────────────────────────────────
JOBS_DIR = Path("/tmp/clipforge/jobs")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# ── Serve frontend ────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")

# ── In-memory job store ───────────────────────────────────────
jobs: dict = {}


# ── Request model ─────────────────────────────────────────────
class ForgeRequest(BaseModel):
    url: str
    detect_audio: bool = True
    detect_transcript: bool = True
    detect_visual: bool = True
    captions: bool = False
    formats: list[str] = ["9:16", "16:9", "1:1"]


# ── Routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ClipForge API running", "version": "1.0"}


@app.post("/forge")
def forge(req: ForgeRequest):
    url = req.url.strip()
    if not re.search(r'(^|[./])youtube\.com', url) and not re.search(r'(^|[./])youtu\.be', url):
        raise HTTPException(status_code=400, detail="URL must be a YouTube link")
    if not req.formats:
        raise HTTPException(status_code=400, detail="Select at least one output format")

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "status": "queued",
        "step": "Job queued, starting...",
        "progress": 0,
        "clips": [],
        "error": None,
        "title": ""
    }
    thread = threading.Thread(target=run_forge_job, args=(job_id, req), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/download/{job_id}/{clip_id}/{fmt}")
def download_clip(job_id: str, clip_id: str, fmt: str):
    path = JOBS_DIR / job_id / "clips" / f"{clip_id}_{fmt}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Clip not found: {path.name}")
    return FileResponse(path=str(path), media_type="video/mp4",
                        filename=f"clipforge_{clip_id}_{fmt}.mp4")


# ── Job state helper ──────────────────────────────────────────

def set_step(job_id: str, step: str, progress: int):
    jobs[job_id]["step"] = step
    jobs[job_id]["progress"] = progress
    jobs[job_id]["status"] = "processing"


# ── Audio energy analysis using only stdlib + ffmpeg ─────────
# No librosa, no numpy, no torch needed

def analyze_audio_energy(wav_path: str, duration: float) -> list:
    """
    Read WAV file with Python stdlib, compute RMS energy per 0.5s window,
    find peaks. No external dependencies beyond stdlib.
    """
    moments = []
    try:
        with wave.open(wav_path, 'rb') as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()

            hop = sr // 2  # 0.5 second windows
            all_rms = []

            # Read in chunks
            wf.rewind()
            frame_idx = 0
            while True:
                raw = wf.readframes(hop)
                if not raw:
                    break
                # Unpack samples
                n_samples = len(raw) // sampwidth // n_channels
                if sampwidth == 2:
                    fmt = f"<{n_samples * n_channels}h"
                    samples = struct.unpack(fmt, raw[:n_samples * n_channels * sampwidth])
                else:
                    frame_idx += 1
                    continue

                # Take first channel only, compute RMS
                mono = samples[::n_channels] if n_channels > 1 else samples
                if len(mono) == 0:
                    frame_idx += 1
                    continue
                rms = (sum(s * s for s in mono) / len(mono)) ** 0.5
                t = frame_idx * hop / sr
                all_rms.append((t, rms))
                frame_idx += 1

            if not all_rms:
                return moments

            # Find 80th percentile threshold
            sorted_rms = sorted(r for _, r in all_rms)
            thresh_idx = int(len(sorted_rms) * 0.80)
            thresh = sorted_rms[thresh_idx]

            # Filter peaks within valid range
            peaks = [
                (t, r) for t, r in all_rms
                if r > thresh and 5.0 < t < duration - 15.0
            ]

            # Deduplicate — keep highest energy, min 20s apart
            peaks.sort(key=lambda x: -x[1])
            last_t = -999.0
            for t, score in peaks:
                if t - last_t > 20.0:
                    moments.append({"time": t, "score": score, "type": "audio"})
                    last_t = t
                if len(moments) >= 6:
                    break

    except Exception as e:
        print(f"[ClipForge] Audio analysis error: {e}")

    return moments


def analyze_transcript(audio_path: str, duration: float) -> list:
    """
    Whisper transcript analysis — lazy imported.
    Skipped cleanly if whisper/torch not installed.
    """
    moments = []
    try:
        import whisper
        model = whisper.load_model("tiny")
        result = model.transcribe(audio_path, fp16=False, verbose=False)
        keywords = [
            "amazing", "incredible", "unbelievable", "never seen",
            "first time", "wait", "can't believe", "oh my", "wow",
            "breaking", "secret", "truth", "insane", "crazy",
            "huge", "biggest", "finally", "shocked", "no way"
        ]
        raw = []
        for seg in result.get("segments", []):
            text_lower = seg["text"].lower()
            score = sum(1 for kw in keywords if kw in text_lower)
            t = float(seg["start"])
            if score > 0 and 5.0 < t < duration - 15.0:
                raw.append({
                    "time": t,
                    "score": float(score) * 0.4,
                    "type": "ai",
                    "text": seg["text"].strip()[:80]
                })
        raw.sort(key=lambda x: -x["score"])
        last_t = -999.0
        for m in raw:
            if m["time"] - last_t > 20.0:
                moments.append(m)
                last_t = m["time"]
            if len(moments) >= 6:
                break
    except Exception as e:
        print(f"[ClipForge] Transcript skipped: {e}")
    return moments


def analyze_visual(video_path: str, duration: float, job_dir: Path) -> list:
    """
    Visual motion detection via frame diffing.
    Uses ffmpeg + opencv (optional) or pure ffmpeg scene detection.
    """
    moments = []
    try:
        # Try opencv first
        import cv2
        import numpy as np
        frames_dir = job_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vf", "fps=1,scale=160:90",
             str(frames_dir / "frame_%06d.jpg"),
             "-loglevel", "quiet"],
            check=True, capture_output=True
        )
        frame_files = sorted(frames_dir.glob("*.jpg"))
        prev = None
        motion_scores = []
        for idx, fp in enumerate(frame_files):
            img = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if prev is not None:
                diff = float(np.mean(
                    np.abs(img.astype(float) - prev.astype(float))
                ))
                motion_scores.append((float(idx), diff))
            prev = img
        shutil.rmtree(str(frames_dir), ignore_errors=True)

        if motion_scores:
            sorted_scores = sorted(s for _, s in motion_scores)
            thresh_idx = int(len(sorted_scores) * 0.85)
            thresh = sorted_scores[thresh_idx]
            last_t = -999.0
            for sec, score in sorted(motion_scores, key=lambda x: -x[1]):
                if (score > thresh and 5.0 < sec < duration - 15.0
                        and sec - last_t > 20.0):
                    moments.append({"time": sec, "score": score / 50.0, "type": "visual"})
                    last_t = sec
                if len(moments) >= 5:
                    break

    except ImportError:
        # opencv not available — use ffmpeg scene detection instead
        try:
            result = subprocess.run(
                ["ffmpeg", "-i", video_path,
                 "-vf", "select='gt(scene,0.3)',showinfo",
                 "-f", "null", "-"],
                capture_output=True, text=True
            )
            last_t = -999.0
            for line in result.stderr.split('\n'):
                if 'pts_time:' in line:
                    try:
                        t = float(line.split('pts_time:')[1].split()[0])
                        if 5.0 < t < duration - 15.0 and t - last_t > 20.0:
                            moments.append({"time": t, "score": 0.6, "type": "visual"})
                            last_t = t
                        if len(moments) >= 5:
                            break
                    except Exception:
                        continue
        except Exception as e:
            print(f"[ClipForge] FFmpeg scene detection failed: {e}")

    except Exception as e:
        print(f"[ClipForge] Visual detection skipped: {e}")

    return moments


# ── Core processing ───────────────────────────────────────────

def run_forge_job(job_id: str, req: ForgeRequest):
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = job_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    video_path: str = ""
    audio_path: str = ""

    try:
        # ── 1. Download ───────────────────────────────────────
        set_step(job_id, "Downloading video from YouTube...", 5)

        ydl_opts = {
            "format": (
                "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]"
                "/bestvideo[height<=1080]+bestaudio"
                "/best[height<=1080]"
                "/best"
            ),
            "outtmpl": str(job_dir / "source.%(ext)s"),
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": 60,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.url.strip(), download=True)
            title: str = info.get("title", "Untitled Video") if info else "Untitled Video"

        for candidate in job_dir.iterdir():
            if candidate.stem == "source" and candidate.suffix in (".mp4", ".mkv", ".webm"):
                video_path = str(candidate)
                break

        if not video_path or not Path(video_path).exists():
            raise RuntimeError(
                "Download failed — no video file found. "
                "The video may be private, age-restricted, or region-blocked."
            )

        set_step(job_id, f"Downloaded: {title[:60]}", 15)

        # ── 2. Probe duration ─────────────────────────────────
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", video_path],
            capture_output=True, text=True
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            raise RuntimeError("ffprobe could not read the video. File may be corrupted.")

        probe_json = json.loads(probe.stdout)
        fmt_info = probe_json.get("format", {})
        duration_str = fmt_info.get("duration", "0")
        duration = float(duration_str) if duration_str else 0.0

        if duration < 30:
            raise RuntimeError(f"Video is only {int(duration)}s — minimum 30 seconds required.")

        # ── 3. Extract audio WAV ──────────────────────────────
        audio_path = str(job_dir / "audio.wav")
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", video_path,
             "-vn", "-acodec", "pcm_s16le", "-ar", "22050", "-ac", "1",
             audio_path, "-loglevel", "quiet"],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            raise RuntimeError(f"Audio extraction failed: {r.stderr[-300:]}")

        # ── 4. Audio peak detection (stdlib only) ─────────────
        audio_moments: list = []
        if req.detect_audio:
            set_step(job_id, "Analyzing audio energy peaks...", 30)
            audio_moments = analyze_audio_energy(audio_path, duration)

        # ── 5. Transcript analysis (optional — needs whisper) ─
        transcript_moments: list = []
        if req.detect_transcript:
            set_step(job_id, "Transcribing speech highlights...", 48)
            transcript_moments = analyze_transcript(audio_path, duration)

        # ── 6. Visual motion detection (optional — needs cv2) ─
        visual_moments: list = []
        if req.detect_visual:
            set_step(job_id, "Detecting visual highlights...", 63)
            visual_moments = analyze_visual(video_path, duration, job_dir)

        # ── 7. Merge + rank ───────────────────────────────────
        set_step(job_id, "Ranking best moments...", 76)
        all_moments = audio_moments + transcript_moments + visual_moments

        # Fallback: evenly spaced if nothing found
        if not all_moments:
            interval = duration / 6.0
            for i in range(5):
                t = interval * (i + 1)
                if 5.0 < t < duration - 15.0:
                    all_moments.append({"time": t, "score": 0.5, "type": "audio"})

        # Normalize scores 60–99
        max_score = max((m["score"] for m in all_moments), default=1.0) or 1.0
        for m in all_moments:
            m["score_pct"] = int((m["score"] / max_score) * 39) + 60

        # Deduplicate — no two within 15s
        all_moments.sort(key=lambda x: -x["score_pct"])
        final_moments: list = []
        used_times: list = []
        for m in all_moments:
            t = m["time"]
            if not used_times or all(abs(t - ut) > 15.0 for ut in used_times):
                final_moments.append(m)
                used_times.append(t)

        # ── 8. Cut clips ──────────────────────────────────────
        set_step(job_id, "Cutting clips...", 82)

        FORMAT_CONFIGS = {
            "9:16": {
                "vf": "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale=1080:1920",
                "suffix": "9-16"
            },
            "16:9": {
                "vf": "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black",
                "suffix": "16-9"
            },
            "1:1": {
                "vf": "crop='min(iw,ih)':'min(iw,ih)',scale=1080:1080",
                "suffix": "1-1"
            },
        }

        CLIP_LENGTHS = {"audio": 35, "ai": 55, "visual": 40}
        clips_out: list = []

        for i, moment in enumerate(final_moments):
            clip_len = float(CLIP_LENGTHS.get(moment["type"], 40))
            start = max(0.0, moment["time"] - 5.0)
            end = min(duration, start + clip_len)
            actual_len = end - start

            if actual_len < 5.0:
                continue

            clip_id = f"clip_{i + 1:02d}"
            clip_info = {
                "id": clip_id,
                "index": i + 1,
                "start": fmt_time(start),
                "end": fmt_time(end),
                "duration": fmt_duration(actual_len),
                "type": moment["type"],
                "score": moment["score_pct"],
                "title": moment.get("text") or gen_title(moment["type"], i + 1),
                "formats": {}
            }

            for fmt_key, fmt_cfg in FORMAT_CONFIGS.items():
                if fmt_key not in req.formats:
                    continue

                out_path = clips_dir / f"{clip_id}_{fmt_cfg['suffix']}.mp4"
                vf = fmt_cfg["vf"]

                if req.captions:
                    safe = (
                        clip_info["title"]
                        .replace("'", "").replace('"', "")
                        .replace(":", "").replace("\\", "")
                        .replace(",", "")[:40]
                    )
                    vf += (
                        f",drawtext=text='{safe}':"
                        "fontsize=36:fontcolor=white:x=(w-tw)/2:y=h-100:"
                        "box=1:boxcolor=black@0.6:boxborderw=8"
                    )

                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", video_path,
                    "-t", str(actual_len),
                    "-vf", vf,
                    "-c:v", "libx264",
                    "-crf", "23",
                    "-preset", "fast",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    "-movflags", "+faststart",
                    "-avoid_negative_ts", "make_zero",
                    str(out_path),
                    "-loglevel", "error"
                ]

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    print(f"[ClipForge] ffmpeg error {clip_id}/{fmt_key}: {proc.stderr[-400:]}")
                    continue

                if out_path.exists() and out_path.stat().st_size > 1000:
                    clip_info["formats"][fmt_key] = (
                        f"/download/{job_id}/{clip_id}/{fmt_cfg['suffix']}"
                    )

            if clip_info["formats"]:
                clips_out.append(clip_info)

        if not clips_out:
            raise RuntimeError(
                "No clips could be created. "
                "This can happen with very short videos or unsupported formats."
            )

        # ── 9. Cleanup ────────────────────────────────────────
        for p in [video_path, audio_path]:
            try:
                if p:
                    Path(p).unlink(missing_ok=True)
            except Exception:
                pass

        jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "step": f"Done! {len(clips_out)} clips ready to download.",
            "clips": clips_out,
            "title": title,
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error",
            "error": str(e),
            "step": f"Error: {str(e)}"
        })
        try:
            shutil.rmtree(str(job_dir), ignore_errors=True)
        except Exception:
            pass


# ── Helpers ───────────────────────────────────────────────────

def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def fmt_duration(seconds: float) -> str:
    if seconds >= 60:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}m {s}s"
    return f"{int(seconds)}s"


def gen_title(clip_type: str, index: int) -> str:
    titles = {
        "audio": [
            "Peak energy moment", "Loudest crowd reaction",
            "High-energy spike", "Explosive moment",
            "Big reaction", "Audio highlight",
        ],
        "ai": [
            "Key insight moment", "Viral-worthy statement",
            "Highlight quote", "Most talked-about moment",
            "Memorable line", "Quotable moment",
        ],
        "visual": [
            "High-motion sequence", "Dynamic visual moment",
            "Fast-paced action", "Visual highlight",
            "Scene change peak", "Most visually intense",
        ],
    }
    opts = titles.get(clip_type, titles["audio"])
    return opts[(index - 1) % len(opts)]
