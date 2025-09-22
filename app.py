import os
import cv2
import time
import numpy as np
import mediapipe as mp
from numpy.linalg import norm
from encode_faces import build_encodings  # gera encodings em memória

# =========================
# Parâmetros do App
# =========================
DATASET_DIR = "dataset"     # pastas por pessoa: dataset/Nome/*.jpg|png
EMB_SIZE = 128              # tamanho do “embedding” (resize para EMB_SIZE x EMB_SIZE)
MODEL_SELECTION = 1         # 0: rostos próximos | 1: rostos mais distantes
MIN_DET_CONF = 0.6          # confiança mínima (MediaPipe FaceDetection)
THRESH = 0.32               # limiar (distância cosseno) – menor é mais parecido
MARGIN = 0.06               # diferença mínima pro 2º melhor (segurança)
MAX_MESH_FACES = 5          # quantos rostos o FaceMesh processa p/landmarks

# =========================
# MediaPipe (landmarks e regiões)
# =========================
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

REGIONS = {
    "Olho Esq": getattr(mp_face_mesh, "FACEMESH_LEFT_EYE", None),
    "Olho Dir": getattr(mp_face_mesh, "FACEMESH_RIGHT_EYE", None),
    "Boca": getattr(mp_face_mesh, "FACEMESH_LIPS", None),
    "Sobr. Esq": getattr(mp_face_mesh, "FACEMESH_LEFT_EYEBROW", None),
    "Sobr. Dir": getattr(mp_face_mesh, "FACEMESH_RIGHT_EYEBROW", None),
    "Contorno": getattr(mp_face_mesh, "FACEMESH_FACE_OVAL", None),
    "Íris Esq": getattr(mp_face_mesh, "FACEMESH_LEFT_IRIS", None),
    "Íris Dir": getattr(mp_face_mesh, "FACEMESH_RIGHT_IRIS", None),
}

def region_centroid(landmarks, pairs, w, h):
    if not pairs:
        return None
    idxs = set()
    for i, j in pairs:
        idxs.add(i); idxs.add(j)
    pts = []
    for idx in idxs:
        if 0 <= idx < len(landmarks):
            lm = landmarks[idx]
            x = int(min(max(0, lm.x * w), w - 1))
            y = int(min(max(0, lm.y * h), h - 1))
            pts.append((x, y))
    if not pts:
        return None
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    return int(sum(xs)/len(xs)), int(sum(ys)/len(ys))

def put_label_with_bg(img, text, x, y, font_scale=0.5, thickness=1):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    cv2.rectangle(img, (x-3, y-th-6), (x+tw+3, y+3), (0,0,0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), thickness, cv2.LINE_AA)

def main():
    # 1) Gera encodings em memória (com centróides)
    encodings, centroids, meta = build_encodings(
        dataset_dir=DATASET_DIR,
        size=EMB_SIZE,
        model_selection=MODEL_SELECTION,
        min_detection_confidence=MIN_DET_CONF
    )
    print(f"[INFO] Pessoas carregadas: {len(encodings)} | Params: {meta}")

    # 2) Inicializa MediaPipe e câmera
    detector = mp_face_detection.FaceDetection(model_selection=MODEL_SELECTION,
                                               min_detection_confidence=MIN_DET_CONF)
    mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                 max_num_faces=MAX_MESH_FACES,
                                 refine_landmarks=True,
                                 min_detection_confidence=MIN_DET_CONF)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera (id=0).")

    prev = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Frame não lido da câmera. Saindo.")
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 3) Detecção + identificação via centróides
            res = detector.process(rgb)
            if res.detections:
                for det in res.detections:
                    bb = det.location_data.relative_bounding_box
                    x = int(bb.xmin * w); y = int(bb.ymin * h)
                    ww = int(bb.width * w); hh = int(bb.height * h)
                    x1 = max(0, x); y1 = max(0, y)
                    x2 = min(w, x + ww); y2 = min(h, y + hh)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    face = rgb[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    emb = cv2.resize(face, (EMB_SIZE, EMB_SIZE)).astype("float32").flatten()
                    nrm = norm(emb)
                    if nrm == 0:
                        continue
                    emb = emb / (nrm + 1e-10)

                    # matching contra centróides
                    best_p, best_d = None, 1.0
                    second_d = 1.0
                    for p, c in centroids.items():
                        d = 1.0 - float(np.dot(emb, c))  # distância cosseno (vetores normalizados)
                        if d < best_d:
                            second_d = best_d
                            best_d = d
                            best_p = p
                        elif d < second_d:
                            second_d = d

                    if best_d <= THRESH and (second_d - best_d) >= MARGIN:
                        label = best_p
                        color = (0, 200, 0)
                    else:
                        label = "Desconhecido"
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} ({best_d:.3f})", (x1, max(15, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # 4) Landmarks + rótulos de regiões (sempre ativos)
            mesh_res = mesh.process(rgb)
            if mesh_res.multi_face_landmarks:
                for lms in mesh_res.multi_face_landmarks:
                    # desenha todos os pontos
                    for lm in lms.landmark:
                        cx = int(min(max(0, lm.x * w), w-1))
                        cy = int(min(max(0, lm.y * h), h-1))
                        cv2.circle(frame, (cx, cy), 1, (255, 255, 0), -1)
                    # rótulos de regiões
                    for name, pairs in REGIONS.items():
                        if not pairs:
                            continue
                        cen = region_centroid(lms.landmark, pairs, w, h)
                        if cen:
                            rx, ry = cen
                            cv2.circle(frame, (rx, ry), 3, (255, 0, 255), -1)
                            put_label_with_bg(frame, name, rx + 6, ry + 6, font_scale=0.45, thickness=1)

            # 5) FPS
            now = time.time()
            fps = 1.0 / (now - prev + 1e-10)
            prev = now
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Reconhecimento Facial (mem-only)", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            mesh.close()
        except Exception:
            pass
        try:
            detector.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
