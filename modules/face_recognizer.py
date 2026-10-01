import os
import pickle

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class FaceRecognizer:
    """
    Stage 4. OpenCV SFace. Turns an aligned face into a 128-d embedding and
    compares it against the embeddings built from dataset/<name>/ images.
    """

    def __init__(self, model_path, match_threshold=0.40):
        self.model = cv2.FaceRecognizerSF.create(model_path, "")
        self.match_threshold = match_threshold
        self.names = []                    # one name per stored embedding
        self.embeddings = np.empty((0, 128), dtype=np.float32)

    # ------------------------------------------------------------------ #
    def embed(self, frame, face_row):
        aligned = self.model.alignCrop(frame, face_row)
        feat = self.model.feature(aligned).flatten().astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-10)

    def identify(self, frame, face_row):
        """Returns (name, similarity). name is 'Unknown' below the threshold."""
        if len(self.embeddings) == 0:
            return "Unknown", 0.0
        emb = self.embed(frame, face_row)
        sims = self.embeddings @ emb               # cosine similarity (all are L2-normalised)

        # best score per person (max over that person's images)
        best_name, best_sim = "Unknown", -1.0
        for name in set(self.names):
            idx = [i for i, n in enumerate(self.names) if n == name]
            s = float(sims[idx].max())
            if s > best_sim:
                best_name, best_sim = name, s

        if best_sim < self.match_threshold:
            return "Unknown", best_sim
        return best_name, best_sim

    # ------------------------------------------------------------------ #
    def build_from_dataset(self, dataset_dir, face_detector, use_flip=True):
        """Read every dataset/<name>/<image>, detect the largest face, store its embedding."""
        names, embs = [], []
        people = sorted(d for d in os.listdir(dataset_dir)
                        if os.path.isdir(os.path.join(dataset_dir, d)))
        if not people:
            print(f"[WARN] No person folders found in {dataset_dir}")

        for person in people:
            folder = os.path.join(dataset_dir, person)
            files = [f for f in sorted(os.listdir(folder)) if f.lower().endswith(IMAGE_EXTS)]
            used = 0
            for fname in files:
                img = cv2.imread(os.path.join(folder, fname))
                if img is None:
                    continue
                variants = [img, cv2.flip(img, 1)] if use_flip else [img]
                for v in variants:
                    faces = face_detector.detect(v)
                    if len(faces) == 0:
                        continue
                    largest = max(faces, key=lambda f: f[2] * f[3])
                    embs.append(self.embed(v, largest))
                    names.append(person)
                    used += 1
            print(f"[INFO] {person}: {len(files)} images -> {used} embeddings")
            if used == 0:
                print(f"[WARN] No face found in any image of '{person}'")

        self.names = names
        self.embeddings = np.array(embs, dtype=np.float32) if embs else np.empty((0, 128), np.float32)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"names": self.names, "embeddings": self.embeddings}, f)
        print(f"[INFO] Saved {len(self.names)} embeddings for "
              f"{len(set(self.names))} people -> {path}")

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.names = data["names"]
        self.embeddings = data["embeddings"]
        print(f"[INFO] Loaded {len(self.names)} embeddings for {len(set(self.names))} people")
