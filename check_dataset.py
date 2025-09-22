import os
import cv2
import mediapipe as mp

DATASET_DIR = "dataset"
MODEL_SELECTION = 1
MIN_DET_CONF = 0.6
SHOW = False  # True para visualizar imagens com bounding box

mp_face_detection = mp.solutions.face_detection

def main():
    if not os.path.isdir(DATASET_DIR):
        print(f"[ERRO] Dataset não encontrado em: {DATASET_DIR}")
        return

    total = 0
    ok = 0
    with mp_face_detection.FaceDetection(model_selection=MODEL_SELECTION,
                                         min_detection_confidence=MIN_DET_CONF) as det:
        for person in sorted(os.listdir(DATASET_DIR)):
            pdir = os.path.join(DATASET_DIR, person)
            if not os.path.isdir(pdir):
                continue

            for fname in sorted(os.listdir(pdir)):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    continue
                total += 1
                path = os.path.join(pdir, fname)
                img = cv2.imread(path)
                if img is None:
                    print(f"[WARN] Não abriu: {person}/{fname}")
                    continue
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = det.process(rgb)
                if res.detections:
                    ok += 1
                    print(f"[OK] Rosto detectado em: {person}/{fname}")
                    if SHOW:
                        h, w, _ = img.shape
                        d = res.detections[0].location_data.relative_bounding_box
                        x = int(d.xmin * w); y = int(d.ymin * h)
                        ww = int(d.width * w); hh = int(d.height * h)
                        cv2.rectangle(img, (x, y), (x+ww, y+hh), (0,255,0), 2)
                        cv2.imshow("check_dataset", img)
                        if cv2.waitKey(0) & 0xFF == 27:
                            cv2.destroyAllWindows()
                            return
                else:
                    print(f"[FAIL] Sem rosto em: {person}/{fname}")

    print(f"\nResumo: {ok}/{total} imagens com rosto detectado.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
