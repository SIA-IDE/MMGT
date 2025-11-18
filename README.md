
<div align="center">

## <b>MMGT</b>: Motion Mask Guided Two-Stage Network for Co-Speech Gesture Video Generation  🎥🎤  
(IEEE TCSVT 2025)

**Siyuan Wang**, **Jiawei Liu**, **Wei Wang**, **Yeying Jin**, **Jinsong Du**, **Zhi Han** ✨

[Paper (TCSVT 2025)](https://doi.org/10.1109/TCSVT.2025.3604109) 📖

## Overview 🧐

<a href="./pipline_1.png">
  <img src="./demo/pipline_1.png" alt="MMGT Pipeline Overview" width="900px">
</a>

</div>

---

**Co-speech gesture video generation** aims to synthesize expressive talking videos from a still portrait and a speech audio track 🎬🎶. However, purely audio-controlled methods often:

- Miss large body and hand motions 🤦‍♂️  
- Struggle to emphasize key motion regions (face, lips, hands, upper body) 🙄  
- Introduce temporal flickering or visual artifacts 💥

**MMGT** addresses these issues with a **motion-mask–guided two-stage framework**:

1. **SMGA** – *Spatial Mask-Guided Audio2Pose* 🎧➡️💃  
   - Converts audio into high-quality **pose videos**  
   - Predicts **motion masks** to highlight regions with significant movement (face, lips, hands, upper body) 🎯

2. **Diffusion-based Video Generator with MM-HAA** – *Motion-Masked Hierarchical Audio Attention* 🎥  
   - A stabilized diffusion video model  
   - Takes **audio, pose, and motion masks** as input  
   - Generates temporally stable, lip-synchronized, and detail-controllable gesture videos 🕺

---

## Demos 🎥👀

<table>
  <tr>
    <td><img src="./demo/007_j4QlG5jKpio_audio_007.gif" width="100%"></td>
    <td><img src="./demo/099_0BF2Np5J6jY_audio_004.gif" width="100%"></td>
  </tr>
  <tr>
    <!-- Note: # must be encoded as %23 -->
    <td><img src="./demo/oliver%23103842_slice18.gif" width="100%"></td>
    <td><img src="./demo/pats.gif" width="100%"></td>
  </tr>
</table>

---

## News 📰

- **2025-09-01**: Our paper  
  **“MMGT: Motion Mask Guided Two-Stage Network for Co-Speech Gesture Video Generation”**  
  has been **accepted** to **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**, 2025. 🎉  
  DOI: **10.1109/TCSVT.2025.3604109** 📚

---

## Release Plan (September 2025) 🗓️

We plan to open-source MMGT around **September 2025**, focusing on the following **four deliverables**:

1. **Video demos** 📽️  
2. **Inference code** *(including long-video support)* 💻  
3. **Training code** 🛠️  
4. **Multi-person & multi-scene model weights** 🤖

---

## Environment ⚙️

We recommend the following setup:

- **Python**: `>= 3.10` 🐍  
- **CUDA**: `= 12.4`  
  (Other versions may work but are not thoroughly tested.) 💻

```bash
conda create -n MMGT python=3.10
conda activate MMGT
pip install -r requirements.txt
```

## Checkpoints 🎯

Pre-trained weights are available on HuggingFace:

* [MMGT Pretrained Weights](https://huggingface.co/addtime/MMGT_pretrained/tree/main)

Download the checkpoints and place them according to the paths specified in the config files under `./configs`.

---

## Inference 🔍

> **Note:** The current implementation supports video lengths of **up to 3.2 seconds** ⏱️.  
> Extended / long-video generation will be released together with the full open-source version 🚀.

### 1. Audio-to-Video (Audio2Videos) 🎧➡️🎥

End-to-end generation from **audio + single image**:

```bash
python scripts/audio2vid.py   -c ./configs/prompts/animation.yaml   --image_path /path/to/your/image.png   --audio_path /path/to/your/audio.wav   --out_dir /path/to/output_dir
```

### 2. Pose-to-Video (Pose2Videos) 💃➡️🎥

If you already have pose and motion-mask videos (e.g., from Stage 1 or other methods), you can directly drive the video generator:

```bash
python scripts/pose2vid.py   -c ./configs/prompts/animation.yaml   --image_path /path/to/img.png   --pose_path /path/to/pose.mp4   --face_mask_path /path/to/face.mp4   --lips_mask_path /path/to/lips.mp4   --hands_mask_path /path/to/hands.mp4   --out_dir ./outputs
```

---

## Training 🏋️‍♂️

### data Preparation, Download, and Preprocessing

For detailed data preparation (including dataset structure, preprocessing scripts, and examples), please refer to the data pipeline of:

> [https://github.com/thuhcsi/S2G-MDDiffusion#-data-preparation](https://github.com/thuhcsi/S2G-MDDiffusion#-data-preparation)

#### Next, run the following processing code:

```bash

python -m scripts.data_preprocess --input_dir "Path to the 512×512 training or test video files processed according to the above procedure"
python data/extract_movment_mask_all.py --input_root "Path to the 512×512 training or test video files processed according to the above procedure"

```

#### dataSET FOR TRAIN PROCESS ONE

##### Extract DWpose npy from the videos

```bash
  |-- data/train/
    |-- keypoints/
    |   |-- 0001.npy
    |   |-- 0002.npy 
    |   |-- 0003.npy
    |   `-- 0004.npy
    |-- audios/
    |   |-- 0001.wav
    |   |-- 0002.wav
    |   |-- 0003.wav
    |   `-- 0004.wav

```

```bash
cd data
python create_dataset.py --extract-baseline --extract-wavlm
cd ..
```

#### dataSET FOR TRAIN PROCESS TWO

```bash
    |--- data/train/
    |    |--- videos
    |    |    |--- chemistry#99999.mp4
    |    |    |--- oliver#88888.mp4
    |    |--- audios
    |    |    |--- chemistry#99999.wav
    |    |    |--- oliver#88888.wav
```

#### The final training data structure is:

```bash
    |--- data/train/
    |    |--- videos
    |    |    |--- chemistry#99999.mp4
    |    |    |--- oliver#88888.mp4
    |    |--- audios
    |    |    |--- chemistry#99999.wav
    |    |    |--- oliver#88888.wav
    |    |--- sep_lips_mask
    |    |    |--- chemistry#99999.mp4
    |    |    |--- oliver#88888.mp4
    |    |--- sep_face_mask
    |    |    |--- chemistry#99999.mp4
    |    |    |--- oliver#88888.mp4
    |    |--- videos_dwpose
    |    |    |--- chemistry#99999.mp4
    |    |    |--- oliver#88888.mp4
    |    |--- audio_emb
    |    |    |--- chemistry#99999.pt
    |    |    |--- oliver#88888.pt
```
#### Import the above dataset paths into a .json file for easy code access.

```bash
python scripts/extract_meta_info_stage1.py -r data/videos -n data
python tool/extract_meta_info_stage2_move_mask.py --root_path data/train --dataset_name my_dataset --meta_info_name data
```

---

<details>
<summary><strong>Train Process 1 – SMGA (Audio2Pose + Motion Masks)</strong></summary>

```bash
accelerate train_a2p.py
```

This stage learns to map raw speech audio to:

* **Pose sequences** 💃  
* **Region-specific motion masks** (face, lips, hands, upper body) 🦸‍♂️

</details>

---

<details>
<summary><strong>Train Process 2 – Diffusion Video Generator (with MM-HAA)</strong></summary>

```bash
accelerate launch train_stage_1.py --config configs/train/stage1.yaml
```
```bash
accelerate launch train_stage_2.py --config configs/train/stage2.yaml
```

This stage fine-tunes the diffusion model to:

* Jointly use **audio**, **poses**, and **motion masks**  
* Produce **synchronized**, **artifact-free** gesture videos 📽️  
* Emphasize large-motion regions through **Motion-Masked Hierarchical Audio Attention (MM-HAA)** 🎯

</details>

---

## Citation 📑

If you find **MMGT** useful in your research, please consider citing our TCSVT 2025 paper:

```bibtex
@ARTICLE{11145152,
  author  = {Wang, Siyuan and Liu, Jiawei and Wang, Wei and Jin, Yeying and Du, Jinsong and Han, Zhi},
  journal = {IEEE Transactions on Circuits and Systems for Video Technology},
  title   = {MMGT: Motion Mask Guided Two-Stage Network for Co-Speech Gesture Video Generation},
  year    = {2025},
  volume  = {},
  number  = {},
  pages   = {1-1},
  keywords= {Videos;Faces;Synchronization;Hands;Lips;Training;Electronic mail;Distortion;data mining;Circuits and systems;Spatial Mask Guided Audio2Pose Generation Network (SMGA);Co-speech Video Generation;Motion Masked Hierarchical Audio Attention (MM-HAA)},
  doi     = {10.1109/TCSVT.2025.3604109}
}
```
