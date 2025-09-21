# Projeto FYORA de reconhecimento facial utilizando as tecnologias OpenCV + MediaPipe

## Objetivo
Este projeto tem como objetivo desenvolver uma aplicação para a FYORA( aplicativo que visa ajudar pessoas viciadas em jogos de apostas) de reconhecimento facial para garantir que a pessoas que estão se cadastrando no aplicativo sejam as mesmas que estão sendo reconhecidas.  
A aplicação utiliza **OpenCV** e **MediaPipe Face Detection/FaceMesh** para:  
- Detectar rostos pela webcam  
- Exibir retângulos de identificação  
- Desenhar landmarks faciais (olhos, boca, sobrancelhas, contorno, etc.)  
- Reconhecer usuários previamente cadastrados a partir de um dataset de imagens  

---

## 🛠️ Tecnologias utilizadas
- [Python 3.12](https://www.python.org/)  
- [OpenCV](https://opencv.org/) → captura de vídeo e processamento de imagens  
- [MediaPipe](https://developers.google.com/mediapipe) → detecção facial e landmarks  
- [NumPy](https://numpy.org/) → operações matriciais e embeddings  
- [Pickle](https://docs.python.org/3/library/pickle.html) → armazenamento de embeddings faciais
- Anaconda navigator

---

## 📂 Estrutura do projeto
```
├── dataset/                # Imagens de referência por pessoa
│   ├── Guilherme/
│   ├── Nikolas/
│   ├── Pedro/
│   ├── Rodrigo/
│   └── Thiago/
│
├── app.py                  # Executa reconhecimento facial em tempo real
├── encode_faces.py         # Processa imagens do dataset e gera embeddings
├── encodings.pkl           # Arquivo gerado com embeddings
├── requirements.txt        # Dependências do projeto
└── README.md               # Documentação do projeto
```

---

## ⚙️ Instalação e execução
### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/Rodrigo-Brasileiro/SprintIotIob
```

### 2️⃣ Criar e ativar ambiente virtual
```bash
python -m venv .venv
.\.venv\Scriptsctivate   # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 4️⃣ Preparar o dataset
Coloque as imagens em `dataset/` com **uma pasta por pessoa**, exemplo:
```
dataset/
 ├── Guilherme/
 │   ├── img1.jpg
 │   └── img2.jpg
 ├── Nikolas/
 ├── Pedro/
 ├── Rodrigo/
 └── Thiago/
```

### 5️⃣ Gerar embeddings
```bash
python encode_faces.py
```

### 6️⃣ Rodar reconhecimento facial
```bash
python app.py
```

Ao rodar, a webcam abrirá e exibirá:  
- Retângulo em torno de cada rosto  
- Nome da pessoa reconhecida ou “Desconhecido”  
- Landmarks faciais (olhos, boca, sobrancelhas, contorno)  
- FPS da execução  

---

## 🎛️ Parâmetros ajustáveis
No código (`app.py` e `encode_faces.py`) alguns parâmetros podem ser modificados:  

- **`min_detection_confidence`** → confiança mínima para detecção (0.3 = mais sensível, 0.8 = mais preciso).  
- **`model_selection`** →  
  - `0`: otimizado para rostos próximos (selfies).  
  - `1`: otimizado para rostos mais distantes.  
- **`size`** → tamanho do embedding (pixels usados para vetorizar o rosto, padrão `128x128`).  
- **`threshold`** → limiar de similaridade (quanto menor, mais estrito para reconhecer).  

## Integrantes
GUILHERME ROCHA BIANCHINI - RM97974
NIKOLAS RODRIGUES MOURA DOS SANTOS - RM551566
PEDRO HENRIQUE PEDROSA TAVARES - RM97877
RODRIGO BRASILEIRO - RM98952
THIAGO JARDIM DE OLIVEIRA - RM551624
