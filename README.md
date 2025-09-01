### Release Plan (Mid-September 2025) — Checklist ✅ (Tokyo time)

- **Sep 1–3 — Repo Prep & Demo** 🔧🎬  
  - [ ] Finalize repo structure, license, and env files (`requirements.txt` / `environment.yml`)  
  - [ ] Record short **video demo** (single image + audio), export **MP4 + GIF**, fixed seed  
  - [ ] Add `demo/` assets and one-click script `demo.sh`

- **Sep 4–7 — Inference Code (incl. long-video)** ⚙️🎞️  
  - [ ] Release `infer.py` with `--image_path` and `--audio_path` + sample assets  
  - [ ] Add **Quickstart** in README with reproducible command  
  - [ ] Implement **long-video** pipeline (chunking + smoothing) in `long_infer.py`  
  - [ ] Smoke test on two envs (CUDA 11/12), document **known issues**

- **Sep 8–11 — Training Code** 🏋️  
  - [ ] Ship minimal reproducible training pipeline: `train.py` + `configs/minimal.yaml`  
  - [ ] Provide small dataset **recipe**, logging, and **intermediate checkpoints**  
  - [ ] Define checkpoint I/O (naming, **SHA256**, simple **download script**)

- **Sep 12–14 — Multi-person & Multi-scene Weights** 🧑‍🤝‍🧑🌆  
  - [ ] Train/curate generalized weights; validate on **2–3 scenarios**  
  - [ ] Publish `weights.md` with **links + checksums** and applicability notes  
  - [ ] Final docs polish; tag **Release Candidate (RC)**

- **Sep 15 — Public Release** 🚀  
  - [ ] Open repository; publish **demo videos**  
  - [ ] Post **changelog** and simple **upgrade path**

> Scope focus: **4 deliverables** — ① Video Demo ② Inference Code (with long-video) ③ Training Code ④ Multi-person & Multi-scene Weights
