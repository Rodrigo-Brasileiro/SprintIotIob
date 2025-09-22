# 👁️ Reconhecimento Facial (Local) – OpenCV + MediaPipe (Mem-Only)

## 🎯 Objetivo
Aplicação **local (desktop/notebook)** que realiza **detecção e identificação facial** em tempo real usando **OpenCV** e **MediaPipe**.  
Tudo funciona **100% em memória** (não grava `encodings.pkl`).

Projeto no contexto da disciplina **IoT & IOB (FIAP – 2025)**.

---

## 🧰 Tecnologias
- Python 3.12
- OpenCV (captura de vídeo e desenho)
- MediaPipe (FaceDetection + FaceMesh)
- NumPy (operações numéricas)

---

## 📂 Estrutura
```
.
├── dataset/                 # Imagens de referência (uma pasta por pessoa)
│   ├── Pedro/
│   │   ├── pedro1.jpg
│   │   └── pedro2.jpg
│   └── Nikolas/
│       ├── niko1.jpg
│       └── niko2.jpg
├── app.py                   # Executa: gera encodings em memória + reconhecimento + landmarks
├── encode_faces.py          # Funções para gerar encodings em memória (sem disco)
├── check_dataset.py         # Verifica se há rostos nas imagens do dataset
├── requirements.txt
└── README.md
```

---

## ⚙️ Como rodar

### 1) Ambiente
```bash
python -m venv .venv
# Windows
.\.venv\Scriptsctivate
# Linux/Mac
source .venv/bin/activate
```

### 2) Dependências
```bash
pip install -r requirements.txt
```

### 3) Dataset (mínimo)
```
dataset/
 ├── PessoaA/
 │   ├── a1.jpg
 │   └── a2.jpg
 └── PessoaB/
     ├── b1.jpg
     └── b2.jpg
```
> Dica: 3–6 fotos por pessoa (frontal e levemente anguladas).

### 4) (Opcional) Validar dataset
```bash
python check_dataset.py
```

### 5) Rodar a aplicação
```bash
python app.py
```
- Abre a webcam.
- Mostra **retângulos** (verde = reconhecido, vermelho = desconhecido).
- Exibe **nome + distância**.
- Desenha **landmarks** (FaceMesh) e **rótulos de regiões** (olhos, boca, sobrancelhas, contorno, íris).
- Mostra **FPS**.
- Pressione **ESC** para sair.

---

## 🎛️ Parâmetros (em `app.py`)
- `EMB_SIZE` (default `128`)  
  Tamanho do recorte redimensionado antes de gerar o “embedding”.  
- `MODEL_SELECTION` (`0` perto | `1` longe)  
  Modelo da FaceDetection do MediaPipe.  
- `MIN_DET_CONF` (default `0.6`)  
  Confiança mínima para detectar rostos.  
- `THRESH` (default `0.32`)  
  Limite de distância cosseno (menor = mais parecido).  
- `MARGIN` (default `0.06`)  
  Diferença mínima entre o melhor e o 2º melhor — reduz confusões.  
- `MAX_MESH_FACES` (default `5`)  
  Quantos rostos o FaceMesh processa para landmarks.

**Impacto prático:**
- Diminuir `MIN_DET_CONF` → detecta mais, porém mais falsos positivos.  
- Aumentar `THRESH` → reconhece com mais facilidade, porém pode errar.  
- Diminuir `THRESH` + manter `MARGIN` → mais estrito (menos erros).  
- `EMB_SIZE` maior → potencialmente mais informação, porém mais custo.

---

## 🧪 Como a identificação funciona
1. O sistema **detecta** o rosto (MediaPipe FaceDetection).  
2. Recorta, redimensiona e **normaliza** para gerar um vetor (embedding) simples.  
3. Compara com os **centróides** (médias normalizadas) de cada pessoa do dataset.  
4. Se a **distância cosseno** do melhor for ≤ `THRESH` **e** distante do 2º melhor por ≥ `MARGIN`, **reconhece**; caso contrário, marca **Desconhecido**.

> Observação: essa abordagem é **didática** e atende ao escopo da disciplina. Para produção, prefira embeddings de modelos pré-treinados (ex.: `face_recognition`/dlib/DeepFace).

---

## 🛠️ Dicas de qualidade do dataset
- Use 3–6 fotos por pessoa (frontal e levemente anguladas).  
- Iluminação razoável (evite sombra extrema).  
- Evite fotos borradas ou muito pequenas.  
- Atualize as fotos e rode novamente o app (ele recalcula em memória a cada execução).

---

## 🧯 Troubleshooting
- **Câmera não abre** → troque o ID em `cv2.VideoCapture(0)` para `1` ou `2`.  
- **Rosto não detecta** → diminua `MIN_DET_CONF` (ex.: 0.4) e valide com `check_dataset.py`.  
- **Reconhece errado** → diminua `THRESH` (ex.: 0.28) e/ou aumente `MARGIN` (ex.: 0.08).  
- **FPS baixo** → comente a parte dos landmarks ou reduza `MAX_MESH_FACES`.

---

## 👥 Integrantes
- **Pedro Oliveira Valotto** — RM 551445  
- **Rony Ken Nagai** — RM 551549  
- **Tomáz Versolato Carballo** — RM 551417
