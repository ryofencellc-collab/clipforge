# ClipForge 🎬
**AI-powered YouTube clipping engine — paste a URL, get viral clips in 9:16 / 16:9 / 1:1**

---

## What it does

1. You paste a YouTube URL
2. ClipForge downloads the video, then runs 3 AI detectors in parallel:
   - **Audio peaks** — finds the loudest, most energetic moments (librosa)
   - **Transcript analysis** — finds viral keywords via Whisper speech-to-text
   - **Visual motion** — detects high-action scene changes (OpenCV)
3. Cuts the best moments into ready-to-post clips in every format
4. Gives you download links for each clip × each format

---

## Deploy on Railway (takes ~10 minutes, free)

### Step 1 — Put the files on GitHub

1. Go to **github.com** → click **New repository**
2. Name it `clipforge`, set it to **Public**, click **Create repository**
3. On the next page, click **uploading an existing file**
4. Upload ALL files maintaining this exact folder structure:

```
clipforge/
├── backend/
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   └── index.html
├── railway.toml
├── nixpacks.toml
└── README.md
```

> **Tip:** You can drag the entire `clipforge` folder from the zip into the GitHub upload page — it preserves structure automatically.

5. Click **Commit changes**

---

### Step 2 — Deploy on Railway

1. Go to **railway.app** → click **Start a New Project**
2. Choose **Deploy from GitHub repo**
3. Authorize Railway to access your GitHub if prompted
4. Select your `clipforge` repo → click **Deploy Now**
5. Railway will start building (you can watch logs in real time)

**First build takes 8–12 minutes** — it's installing PyTorch, Whisper, OpenCV etc. Subsequent deploys are faster.

---

### Step 3 — Get your URL

1. Once the build is green, click **Settings** → **Networking** → **Generate Domain**
2. Copy your domain (looks like `clipforge-production-xxxx.up.railway.app`)
3. Open: **`https://YOUR-DOMAIN/app`**

That's your ClipForge app — bookmark it.

---

### Step 4 — Test it works

1. Open `/app` in your browser
2. Paste any public YouTube URL (try a 5–10 min video for best results)
3. Select your formats and hit **⚡ FORGE CLIPS**
4. You'll see the progress bar update every 1.5 seconds
5. When done, click the download buttons to save your clips

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Build fails at pip install | Check Railway build logs — usually a timeout, just redeploy |
| "Video not found" error | Video is private, age-restricted, or region-blocked |
| "Video too short" error | Minimum 30 seconds required |
| Download link gives 404 | Clip may have been cleaned up — re-run the job |
| App loads but API errors | Check the Railway deployment logs for Python errors |

---

## File structure explained

```
clipforge/
├── backend/
│   ├── main.py          # FastAPI server — all processing logic lives here
│   └── requirements.txt # Python dependencies (pinned versions)
├── frontend/
│   └── index.html       # Single-file UI — no build step needed
├── railway.toml         # Railway deploy settings (healthcheck, restart policy)
├── nixpacks.toml        # Nix build config — installs ffmpeg + python3.11
└── README.md            # This file
```

---

## How to use for your clipping business

**Model 1 — Sell clips to small creators**
- Find creators with 1K–50K subscribers in your niche
- DM them: *"I made 5 clips from your [video title] — want to see them?"*
- Forge the clips first, then send 1 as a free sample
- Charge $15–50/video for the full set

**Model 2 — Revenue share**
- Clone trending content niches (finance, fitness, gaming highlights)
- Post 3–5 clips per day across TikTok/Shorts/Reels
- Monetize with YouTube Partner Program + TikTok Creator Fund

---

## Technical notes

- Free Railway plan: 500 hrs/month — enough for ~300 videos/month
- Processing time: 2–8 min per video (depends on length + which detectors run)
- Clips are stored in `/tmp` — Railway ephemeral storage, auto-wiped on restart
- The app runs a single uvicorn worker — fine for personal/small-team use
- Job status is in-memory — restarting the server clears active jobs
