import os
import cv2
import numpy as np
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection

def extract_face_embedding(image_path, size=128, model_selection=1, min_detection_confidence=0.6):
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(model_selection=model_selection,
                                         min_detection_confidence=min_detection_confidence) as det:
        res = det.process(rgb)
    if not res.detections:
        return None

    d = res.detections[0].location_data.relative_bounding_box
    h, w, _ = img.shape
    x = max(int(d.xmin * w), 0)
    y = max(int(d.ymin * h), 0)
    ww = max(int(d.width * w), 0)
    hh = max(int(d.height * h), 0)
    x2 = min(x + ww, w)
    y2 = min(y + hh, h)
    if x >= x2 or y >= y2:
        return None

    face = rgb[y:y2, x:x2]
    if face.size == 0:
        return None

    emb = cv2.resize(face, (size, size)).astype("float32").flatten()
    nrm = np.linalg.norm(emb)
    if nrm == 0:
        return None
    return emb / (nrm + 1e-10)

def build_encodings(dataset_dir="dataset", size=128, model_selection=1, min_detection_confidence=0.6):
    """
    Lê dataset/<Pessoa>/*.jpg|png, gera:
      - encodings: { pessoa: [emb1, emb2, ...] }
      - centroids: { pessoa: media_normalizada }
      - meta: parâmetros usados
    Tudo em memória (não grava arquivos).
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset não encontrado em: {dataset_dir}")

    encodings = {}
    total_imgs = 0
    ok_imgs = 0

    for person in sorted(os.listdir(dataset_dir)):
        pdir = os.path.join(dataset_dir, person)
        if not os.path.isdir(pdir):
            continue

        person_embs = []
        for fname in sorted(os.listdir(pdir)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                continue
            total_imgs += 1
            path = os.path.join(pdir, fname)
            emb = extract_face_embedding(path, size=size,
                                         model_selection=model_selection,
                                         min_detection_confidence=min_detection_confidence)
            if emb is not None:
                person_embs.append(emb)
                ok_imgs += 1
            else:
                print(f"[WARN] Sem rosto válido em: {person}/{fname}")

        if person_embs:
            encodings[person] = person_embs

    if not encodings:
        raise RuntimeError("Nenhum embedding válido foi gerado. Verifique o dataset.")

    centroids = {}
    for person, embs in encodings.items():
        arr = np.vstack(embs).astype(np.float32)
        c = arr.mean(axis=0)
        c /= (np.linalg.norm(c) + 1e-10)
        centroids[person] = c

    meta = {
        "size": size,
        "model_selection": model_selection,
        "min_det_conf": min_detection_confidence,
        "total_imagens": total_imgs,
        "imagens_ok": ok_imgs
    }
    print(f"[OK] Encodings em memória: {len(encodings)} pessoas | imagens OK {ok_imgs}/{total_imgs}")
    return encodings, centroids, meta
