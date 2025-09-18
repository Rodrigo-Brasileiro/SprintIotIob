import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()

base = "dataset"
faces, ids, labels = [], [], []
label_names = []

for i, person in enumerate(os.listdir(base)):
    path = os.path.join(base, person)
    if not os.path.isdir(path):
        continue
    label_names.append(person)
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (200, 200))
        faces.append(img)
        ids.append(i)

recognizer.train(faces, np.array(ids))
recognizer.save("face_model.yml")
with open("labels.txt", "w") as f:
    for name in label_names:
        f.write(name + "\n")

print("Modelo treinado e salvo.")
