"""
Builds face embeddings from dataset/<name>/ images and saves them to models/embeddings.pkl.
Run this after adding or removing people. main.py also runs it automatically
when the dataset folder has changed since the last build.

    python build_embeddings.py
"""
import config
from modules.face_detector import FaceDetector
from modules.face_recognizer import FaceRecognizer
from modules.utils import download_model, ensure_dirs

ensure_dirs(config.DATASET_DIR, config.MODELS_DIR)
download_model(config.YUNET_URL, config.YUNET_PATH)
download_model(config.SFACE_URL, config.SFACE_PATH)

# lower score threshold for enrolment photos so slightly turned faces are still used
face_detector = FaceDetector(config.YUNET_PATH, score_threshold=0.7,
                             nms_threshold=config.FACE_NMS_THRESHOLD, min_face_size=30)
recognizer = FaceRecognizer(config.SFACE_PATH, config.MATCH_THRESHOLD)

recognizer.build_from_dataset(config.DATASET_DIR, face_detector, config.USE_FLIP_AUGMENTATION)
recognizer.save(config.EMBEDDINGS_FILE)
