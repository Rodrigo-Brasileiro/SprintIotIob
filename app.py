import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from numpy.linalg import norm

# =============================
# Carregar embeddings salvos
# =============================
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

# compatibilidade: pode vir com chave 'encodings'
if isinstance(data, dict) and "encodings" in data:
    encodings = data["encodings"]
    emb_size = data.get("size", 128)
else:
    encodings = data
    emb_size = 128

# Sanitiza os embeddings (garante float32 normalizado)
for person in list(encodings.keys()):
    good = []
    for ref in encodings[person]:
        arr = np.asarray(ref, dtype=np.float32).flatten()
        if arr.size == 0:
            continue
        nrm = np.linalg.norm(arr)
        if nrm == 0:
            continue
        good.append(arr / (nrm + 1e-10))
    encodings[person] = good

print(f"[INFO] Carregadas {len(encodings)} pessoas com embeddings (size={emb_size})")

# =============================
# Inicializa MediaPipe
# =============================
mp_face_detection = mp.solutions.face_detection
mp_face_mesh_mod = mp.solutions.face_mesh

detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)
face_mesh = mp_face_mesh_mod.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Constantes para regiões faciais
region_constants = {
    "Olho Esq": mp_face_mesh_mod.FACEMESH_LEFT_EYE,
    "Olho Dir": mp_face_mesh_mod.FACEMESH_RIGHT_EYE,
    "Boca": mp_face_mesh_mod.FACEMESH_LIPS,
    "Sobr. Esq": mp_face_mesh_mod.FACEMESH_LEFT_EYEBROW,
    "Sobr. Dir": mp_face_mesh_mod.FACEMESH_RIGHT_EYEBROW,
    "Contorno": mp_face_mesh_mod.FACEMESH_FACE_OVAL,
    "Íris Esq": getattr(mp_face_mesh_mod, "FACEMESH_LEFT_IRIS", None),
    "Íris Dir": getattr(mp_face_mesh_mod, "FACEMESH_RIGHT_IRIS", None),
}

def cosine_distance(a, b):
    return 1.0 - float(np.dot(a, b))

def region_centroid_from_indices(landmarks, indices, img_w, img_h):
    pts = []
    for idx in indices:
        if 0 <= idx < len(landmarks):
            lm = landmarks[idx]
            x = int(lm.x * img_w)
            y = int(lm.y * img_h)
            pts.append((x, y))
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

# =============================
# Loop da câmera
# =============================
cap = cv2.VideoCapture(0)
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detector.process(rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)

            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + w_box), min(h, y + h_box)

            face_rgb = rgb[y1:y2, x1:x2]
            if face_rgb.size == 0:
                continue

            # Embedding do rosto atual
            emb = cv2.resize(face_rgb, (emb_size, emb_size)).flatten().astype(np.float32)
            nrm = np.linalg.norm(emb)
            if nrm == 0:
                continue
            emb = emb / (nrm + 1e-10)

            # Matching com banco
            best_person, best_score = None, 1.0
            for person, refs in encodings.items():
                for ref in refs:
                    if emb.shape != ref.shape:
                        continue
                    dist = cosine_distance(emb, ref)
                    if dist < best_score:
                        best_score = dist
                        best_person = person

            label = best_person if (best_person and best_score <= 0.40) else "Desconhecido"
            color = (0, 255, 0) if label != "Desconhecido" else (0, 0, 255)

            # Desenha retângulo e identificação
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} ({best_score:.3f})", (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Landmarks e regiões
    mesh_res = face_mesh.process(rgb)
    if mesh_res.multi_face_landmarks:
        for face_landmarks in mesh_res.multi_face_landmarks:
            # Pontos individuais
            for lm in face_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 1, (255, 255, 0), -1)

            # Rótulos de regiões
            for region_name, const_pairs in region_constants.items():
                if not const_pairs:
                    continue
                indices = set()
                for pair in const_pairs:
                    indices.add(pair[0])
                    indices.add(pair[1])
                centroid = region_centroid_from_indices(face_landmarks.landmark, indices, w, h)
                if centroid:
                    rx, ry = centroid
                    cv2.circle(frame, (rx, ry), 3, (255, 0, 255), -1)
                    cv2.putText(frame, region_name, (rx + 5, ry + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # FPS
    cur_time = time.time()
    fps = 1.0 / (cur_time - prev_time + 1e-10)
    prev_time = cur_time
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    cv2.imshow("Reconhecimento Facial Completo", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
