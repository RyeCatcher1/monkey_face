# monkey_matcher_clip.py

import cv2
import torch
import clip
import numpy as np
from pathlib import Path
from PIL import Image
import os
import time

# -----------------------------
# DEVICE
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# LOAD CLIP
# -----------------------------
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# -----------------------------
# LOAD MONKEY IMAGES
# -----------------------------
monkey_dir = Path("monkeys")
if not monkey_dir.exists():
    raise FileNotFoundError("Folder 'monkeys' does not exist")

monkey_names = []
monkey_images = []
monkey_embeddings = []

for img_path in sorted(monkey_dir.iterdir()):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    monkey_names.append(img_path.stem)
    monkey_images.append(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tensor = preprocess(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(img_tensor).cpu().numpy()

    emb /= np.linalg.norm(emb)
    monkey_embeddings.append(emb)

monkey_embeddings = np.vstack(monkey_embeddings)

# -----------------------------
# LOAD / INIT PROTOTYPES
# -----------------------------
proto_file = "prototypes.npy"

if os.path.exists(proto_file):
    prototypes = np.load(proto_file)
    print("LOADED EXISTING PROTOTYPES")
else:
    prototypes = monkey_embeddings.copy()
    print("INITIALIZED PROTOTYPES")

prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True)

# -----------------------------
# LEARNING
# -----------------------------
LR = 0.05

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

print("KEYS:")
print("  0 -> monkey 0")
print("  1..9 -> monkey 1..9")
print("  ESC -> exit")

last_feedback = ""
last_feedback_time = 0

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    frame_small = cv2.resize(frame, (320, 320))
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    frame_tensor = preprocess(frame_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        emb = model.encode_image(frame_tensor).cpu().numpy()

    emb /= np.linalg.norm(emb)

    sims = (emb @ prototypes.T)[0]
    order = np.argsort(-sims)
    winner_idx = order[0]

    # ---- scores overlay ----
    y = 30
    for idx in order:
        txt = f"{idx}: {monkey_names[idx]}  {sims[idx]:+.3f}"
        cv2.putText(display, txt, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 28

    # ---- feedback overlay ----
    if time.time() - last_feedback_time < 1.5:
        cv2.putText(display, last_feedback, (20, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("CAMERA + SCORES", display)

    # ---- winner image ----
    winner_img = cv2.resize(monkey_images[winner_idx], (400, 400))
    cv2.imshow("WINNER MONKEY", winner_img)

    # ---- key handling ----
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break

    if ord('0') <= key <= ord('9'):
        target_idx = key - ord('0')

        if target_idx < len(monkey_names):
            prototypes[target_idx] = (
                (1 - LR) * prototypes[target_idx] + LR * emb
            )
            prototypes[target_idx] /= np.linalg.norm(prototypes[target_idx])

            np.save(proto_file, prototypes)

            last_feedback = f"UPDATED PROTOTYPE: {monkey_names[target_idx]}"
            last_feedback_time = time.time()

            print(last_feedback)

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
