# --------------------------------------------------------------
#  src/pipeline_mediapipe.py
# --------------------------------------------------------------
import cv2, csv, os, time, matplotlib.pyplot as plt, numpy as np
import mediapipe as mp
from collections import Counter

# ================= CONFIG =================
# --- project root (the folder that contains data/, src/, CRICKET_ANALYSIS_RESULTS/) ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

INPUT_VIDEO   = os.path.join(PROJECT_ROOT, "data", "net_session.mp4")
OUTPUT_DIR    = os.path.join(PROJECT_ROOT, "CRICKET_ANALYSIS_RESULTS")
CSV_OUT       = os.path.join(OUTPUT_DIR, "batsman_stance.csv")
PLOT_OUT      = os.path.join(OUTPUT_DIR, "stance_plot.png")
VIDEO_OUT     = os.path.join(OUTPUT_DIR, "batsman_labeled.mp4")

# ROI (fraction of full frame) – keep the values you liked
ROI_X1, ROI_Y1, ROI_X2, ROI_Y2 = 0.40, 0.25, 0.60, 0.75

FRAME_HISTORY = 15          # smoothing window
# --------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= MediaPipe SETUP =================
print("Loading MediaPipe Pose...")
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2,
)

# ================= VIDEO INPUT =================
cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video at:\n   {INPUT_VIDEO}")

fps   = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_vid = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (width, height))

print(f"Video: {width}×{height} @ {fps:.1f} fps – {total_frames} frames")

# ================= HELPERS =================
def remap(landmarks, crop_x, crop_y, crop_w, crop_h):
    """Map ROI coordinates back to full-frame coordinates."""
    return [
        type('lm', (), {
            'x': (lm.x * crop_w + crop_x) / width,
            'y': (lm.y * crop_h + crop_y) / height,
            'visibility': getattr(lm, 'visibility', 1.0)
        }) for lm in landmarks
    ]

def get_stance(landmarks):
    """Simple heuristic – you can expand it later."""
    try:
        # MediaPipe indices: 11-left_shoulder,12-right_shoulder,13-left_elbow,14-right_elbow,
        # 15-left_wrist,16-right_wrist,23-left_hip,24-right_hip
        ls, rs, le, re, lw, rw, lh, rh = [landmarks[i] for i in [11,12,13,14,15,16,23,24]]

        score = {'right':0, 'left':0}

        # Shoulder / hip horizontal offset
        if rs.x - ls.x > 0: score['right'] += 2
        else:               score['left']  += 2

        # Hand vertical position (lower wrist = top hand)
        if lw.y > rw.y: score['right'] += 2
        else:           score['left']  += 2

        return "Right-Handed" if score['right'] >= score['left'] else "Left-Handed"
    except Exception:
        return "Right-Handed"   # safe fallback

# ================= MAIN LOOP =================
recent_stances = []
results        = []
frame_idx      = 0
start_time     = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ---- ROI crop ----
    x1 = int(ROI_X1 * width)
    y1 = int(ROI_Y1 * height)
    x2 = int(ROI_X2 * width)
    y2 = int(ROI_Y2 * height)
    roi_rgb = rgb[y1:y2, x1:x2]

    # ---- Pose ----
    res = pose.process(roi_rgb)
    stance = "Right-Handed"                 # default
    if res.pose_landmarks:
        mapped = remap(res.pose_landmarks.landmark, x1, y1, x2-x1, y2-y1)
        stance = get_stance(mapped)

    # ---- Smoothing ----
    recent_stances.append(stance)
    if len(recent_stances) > FRAME_HISTORY:
        recent_stances.pop(0)
    final_stance = Counter(recent_stances).most_common(1)[0][0]

    # ---- Draw ----
    color = (0,255,0) if "Right" in final_stance else (0,0,255)
    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,120,50), 2)
    cv2.putText(frame, f"BATSMAN: {final_stance}",
                (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    out_vid.write(frame)
    results.append((round(frame_idx/fps, 1), final_stance))

    if frame_idx % 600 == 0:
        elapsed = time.time() - start_time
        print(f"Processed {frame_idx}/{total_frames} "
              f"({frame_idx/total_frames*100:.1f}%) – {frame_idx/elapsed:.2f} fps")

cap.release()
out_vid.release()

# ================= SAVE CSV =================
with open(CSV_OUT, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["time_sec", "stance"])
    w.writerows(results)
print("\nCSV →", CSV_OUT)

# ================= PLOT =================
if results:
    times, numeric = zip(*[(t, 1 if "Right" in s else 0) for t, s in results])
    plt.figure(figsize=(12,4))
    plt.plot(times, numeric, color='green', linewidth=1.5)
    plt.yticks([0,1], ["Left-Handed","Right-Handed"])
    plt.xlabel("Time (s)")
    plt.title("Batsman Stance Detection Timeline")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=150)
    plt.close()
    print("Plot →", PLOT_OUT)

print(f"\nFinished – {len(results)} frames processed")