<div align="center">

## <b>MMGT</b>: Motion Mask Guided Two-Stage Network for Co-Speech Gesture Video Generation  
(IEEE TCSVT 2025)

**Siyuan Wang**, **Jiawei Liu**, **Wei Wang**, **Yeying Jin**, **Jinsong Du**, **Zhi Han**

[Paper (TCSVT 2025)](https://doi.org/10.1109/TCSVT.2025.3604109)

</div>

---

## Overview

<a href="./pipline_1.png">
  <img src="./demo/pipline_1.png" alt="MMGT Pipeline Overview" width="900px">
</a>

**Co-speech gesture video generation** aims to synthesize expressive talking videos from a still portrait and a speech audio track. However, purely audio-controlled methods often:

- Miss large body and hand motions  
- Struggle to emphasize key motion regions (face, lips, hands, upper body)  
- Introduce temporal flickering or visual artifacts  

**MMGT** addresses these issues with a **motion-mask–guided two-stage framework**:

1. **SMGA** – *Spatial Mask-Guided Audio2Pose*  
   - Converts audio into high-quality **pose videos**  
   - Predicts **motion masks** to highlight regions with significant movement (face, lips, hands, upper body)

2. **Diffusion-based Video Generator with MM-HAA** – *Motion-Masked Hierarchical Audio Attention*  
   - A stabilized diffusion video model  
   - Takes **audio, pose, and motion masks** as input  
   - Generates temporally stable, lip-synchronized, and detail-controllable gesture videos

---

## Demos

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

## News

- **2025-09-01**: Our paper  
  **“MMGT: Motion Mask Guided Two-Stage Network for Co-Speech Gesture Video Generation”**  
  has been **accepted** to **IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)**, 2025.  
  DOI: **10.1109/TCSVT.2025.3604109**

---

## Release Plan (September 2025)

We plan to open-source MMGT around **September 2025**, focusing on the following **four deliverables**:

1. **Video demos**  
2. **Inference code** *(including long-video support)*  
3. **Training code**  
4. **Multi-person & multi-scene model weights**  

---

## Environment

We recommend the following setup:

- **Python**: `>= 3.10`  
- **CUDA**: `= 12.4`  
  (Other versions may work but are not thoroughly tested.)

```bash
conda create -n MMGT python=3.10
conda activate MMGT
pip install -r requirements.txt
