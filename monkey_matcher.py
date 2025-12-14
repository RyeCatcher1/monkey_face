import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

# ---------- MODELS ----------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---------- LOAD MONKEY EMBEDDINGS ----------
monkey_dir = Path("monkeys")
monkey_images = []
monkey_names = []
monkey_embeddings = []

for img_path in monkey_dir.iterdir():
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    face = mtcnn(img)
    if face is None:
        print("monkey face is None")
        continue

    with torch.no_grad():
        emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()

    emb /= np.linalg.norm(emb)
    monkey_embeddings.append(emb[0])
    monkey_images.append(img)
    monkey_names.append(img_path.name)

monkey_embeddings = np.array(monkey_embeddings)

print(f"Loaded {len(monkey_embeddings)} monkey faces")
print(monkey_embeddings,"monkey_embeddings")

# ---------- WEBCAM LOOP ----------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(frame_rgb)

    if face is not None:
        with torch.no_grad():
            emb = resnet(face.unsqueeze(0).to(device)).cpu().numpy()
        emb /= np.linalg.norm(emb)

        sims = cosine_similarity(emb, monkey_embeddings)
        idx = np.argmax(sims)

        name = monkey_names[idx]
        # sims shape: (1, N)
        scores = sims[0]

        # sort indices by descending similarity
        order = np.argsort(scores)[::-1]

        y0 = 40
        dy = 30

        for rank, i in enumerate(order):
            name = monkey_names[i]
            score = scores[i]

            text = f"{name}: {score:.2f}"

            cv2.putText(
                frame,
                text,
                (20, y0 + rank * dy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    else:
        print("face is none")
    cv2.imshow("Monkey Matcher", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
