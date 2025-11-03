# Sportzengage
# 🏏 Cricket Batsman Stance Detection using MediaPipe Pose

### 🎯 Automatically detect whether a batsman is **Right-Handed** or **Left-Handed** using Pose Estimation.

---

## 📘 Project Overview

This project uses **MediaPipe Pose** to analyze cricket net session videos and determine the batsman’s stance (Right-Handed or Left-Handed).  
It processes every frame of the video, detects human keypoints (shoulders, wrists, hips), applies geometric logic to classify stance, and generates visual + statistical outputs.

---

## 📂 Project Structure


---

## ⚙️ How It Works

1. **ROI (Region of Interest)**
   - The frame section where the batsman stands is defined:
     ```python
     ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 0.40, 0.25, 0.60, 0.75
     ```
   - This filters out the bowler or background, focusing only on the batting area.

2. **Pose Detection**
   - Uses **MediaPipe Pose** to detect key body points.
   - Tracks shoulders, wrists, elbows, and hips for each frame.

3. **Stance Classification**
   - Based on landmark geometry:
     ```python
     if right_shoulder.x > left_shoulder.x:
         score_right += 2
     if left_wrist.y > right_wrist.y:
         score_right += 2
     ```
   - ✅ Right-Handed → Left hand lower + Right shoulder forward  
   - ✅ Left-Handed → Right hand lower + Left shoulder forward

4. **Frame Smoothing**
   - Uses a 15-frame history to reduce flickering:
     ```python
     recent_stances.append(stance)
     if len(recent_stances) > FRAME_HISTORY:
         recent_stances.pop(0)
     final_stance = Counter(recent_stances).most_common(1)[0][0]
     ```

5. **Outputs**
   - Draws ROI and stance label on each frame
   - Saves final results as:
     - `batsman_labeled.mp4`
     - `batsman_stance.csv`
     - `stance_plot.png`

---

## 🧠 Key Logic

| Condition | Interpretation |
|------------|----------------|
| `right_shoulder.x > left_shoulder.x` | Right-hand stance (shoulder forward) |
| `left_wrist.y > right_wrist.y` | Left hand lower → Right-handed |
| `right_wrist.y > left_wrist.y` | Right hand lower → Left-handed |
| Fallback | Default to Right-Handed |

---

## 🖥️ How to Run (Windows)

### Step 1: Open PowerShell in the project folder
```powershell
cd "C:\Users\divya\OneDrive\Documents\SportzEngage"


Step 2: Set up environment
python -m venv venv
.\venv\Scripts\activate
pip install opencv-python mediapipe matplotlib numpy

Step 3: Run the script
python src\pipeline_mediapipe.py

Step 4: Check results

Outputs will appear in:

C:\Users\divya\OneDrive\Documents\SportzEngage\CRICKET_ANALYSIS_RESULTS

📊 Output Files
File	Description
batsman_stance.csv	Frame-wise stance with timestamps
batsman_labeled.mp4	Video showing detected stance in real time
stance_plot.png	Plot of stance across video duration
🧾 Sample Console Output
Loading MediaPipe Pose...
Video: 1080×1920 @ 30.0 fps – 18007 frames
Processed 600/18007 (3.3%) – 21.5 fps
...
CSV → CRICKET_ANALYSIS_RESULTS\batsman_stance.csv
Plot → CRICKET_ANALYSIS_RESULTS\stance_plot.png
Finished – 18007 frames processed

📈 Example Graph

Y = 1 → Right-Handed

Y = 0 → Left-Handed

X = Time (seconds)

If the graph is mostly green at 1, the batsman is consistently right-handed.

🧩 Algorithm Summary
Step	Process	Output
1	Read video frame	RGB frame
2	Crop to ROI	Focused on batsman
3	Run MediaPipe Pose	Extract keypoints
4	Apply geometric rules	Determine stance
5	Smooth stance history	Reduce noise
6	Draw overlays & save	CSV, MP4, Plot
✅ Why MediaPipe?
Feature	Benefit
Lightweight	Runs real-time on CPU
No GPU required	Works on any laptop
Accurate	Detects small motions
Easy integration	Simple Python API
