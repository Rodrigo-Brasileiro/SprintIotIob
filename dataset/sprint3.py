import cv2
import os


video = cv2.VideoCapture(0)  


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


facemark = cv2.face.createFacemarkLBF()

facemark.loadModel("lbfmodel.yaml")

# === IDENTIFICADOR (LBPH) ===
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Se já tiver um modelo treinado, carregue-o
if os.path.exists("face_model.yml"):
    recognizer.read("face_model.yml")
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f]
else:
    recognizer = None
    labels = []

# === PARÂMETROS AJUSTÁVEIS ===
scaleFactor = 1.1
minNeighbors = 5
minSize = (60, 60)

while True:
    check, img = video.read()
    if not check:
        break

    img = cv2.resize(img, (1280, 720))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # === DETECÇÃO DE FACES ===
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize)

    for (x, y, w, h) in faces:
        # Retângulo
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # === LANDMARKS ===
        _, landmarks = facemark.fit(gray, faces)
        for lm in landmarks:
            for (x_point, y_point) in lm[0]:
                cv2.circle(img, (int(x_point), int(y_point)), 2, (0, 0, 255), -1)

        # === IDENTIFICAÇÃO (se modelo existir) ===
        if recognizer is not None:
            face_roi = gray[y:y + h, x:x + w]
            face_resized = cv2.resize(face_roi, (200, 200))
            label_id, confidence = recognizer.predict(face_resized)
            name = labels[label_id] if confidence < 80 else "Desconhecido"
            cv2.putText(img, f"{name} ({int(confidence)})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Mostrar tela
    cv2.imshow("Deteccao Facial - Haar + Landmarks + ID", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
