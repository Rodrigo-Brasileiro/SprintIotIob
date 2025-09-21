# encode_faces.py
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
import argparse

mp_face_detection = mp.solutions.face_detection

def extract_face_embedding(image_path, model_selection=1, min_detection_confidence=0.6, size=128):
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with mp_face_detection.FaceDetection(model_selection=model_selection,
                                         min_detection_confidence=min_detection_confidence) as detector:
        results = detector.process(rgb)
        if not results.detections:
            return None

        # pega a primeira detecção
        d = results.detections[0].location_data.relative_bounding_box
        h, w, _ = img.shape
        x = max(int(d.xmin * w), 0)
        y = max(int(d.ymin * h), 0)
        w_box = max(int(d.width * w), 0)
        h_box = max(int(d.height * h), 0)

        # ajustar limites
        x2 = min(x + w_box, w)
        y2 = min(y + h_box, h)
        if x >= x2 or y >= y2:
            return None

        face = rgb[y:y2, x:x2]
        if face.size == 0:
            return None

        face_resized = cv2.resize(face, (size, size)).astype("float32").flatten()
        norm = np.linalg.norm(face_resized)
        if norm == 0:
            return None
        return face_resized / norm

def main(args):
    dataset_dir = args.dataset
    out_file = args.output
    encodings = {}  # person -> list of embeddings

    if not os.path.isdir(dataset_dir):
        print(f"[ERRO] dataset não encontrado em: {dataset_dir}")
        return

    for person in sorted(os.listdir(dataset_dir)):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue

        embeddings = []
        for fname in sorted(os.listdir(person_dir)):
            path = os.path.join(person_dir, fname)
            emb = extract_face_embedding(path,
                                         model_selection=args.model_selection,
                                         min_detection_confidence=args.min_detection_confidence,
                                         size=args.size)
            if emb is not None:
                embeddings.append(emb)
            else:
                print(f"[WARN] {person}/{fname} -> rosto não detectado ou inválido")

        if embeddings:
            encodings[person] = embeddings
            print(f"[INFO] {person}: {len(embeddings)} embeddings gerados")

    if not encodings:
        print("[ERRO] Nenhum embedding gerado. Verifique as imagens do dataset.")
        return

    with open(out_file, "wb") as f:
        pickle.dump({"encodings": encodings, "size": args.size}, f)
    print(f"[OK] Encodings salvos em: {out_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gerar encodings a partir do dataset")
    p.add_argument("--dataset", "-d", default="dataset", help="Pasta do dataset (pasta por pessoa)")
    p.add_argument("--output", "-o", default="encodings.pkl", help="Arquivo de saída")
    p.add_argument("--size", type=int, default=128, help="Tamanho do embedding (px). Usa face resized para size x size")
    p.add_argument("--model_selection", type=int, choices=[0,1], default=1, help="MediaPipe model_selection (0 ou 1)")
    p.add_argument("--min_detection_confidence", type=float, default=0.6, help="Confiança mínima para detecção")
    args = p.parse_args()
    main(args)
