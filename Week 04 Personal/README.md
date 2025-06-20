# Bird vs Drone Classification using Transfer Learning (MobileNetV2)

This project is part of a machine learning assignment to classify two image classes — **Bird** and **Drone** — using **Transfer Learning** with **MobileNetV2** pretrained on ImageNet.

---

## 📁 Dataset

- Source: [Birds vs Drone Dataset - Kaggle](https://www.kaggle.com/datasets/harshwalia/birds-vs-drone-dataset)
- Classes: `bird/`, `drone/`
- Total images: 828 (400 bird, 428 drone)
- Split: 80% training, 20% validation

---

## 🧠 Model

- Pretrained model: **MobileNetV2**
- Strategy: Feature extraction + **fine-tuning 20 last layers**
- Input size: 224x224 pixels
- Output: Binary classification (sigmoid)

---

## ⚙️ Training Details

- Epochs: 20 (10 pretrain + 10 fine-tune)
- Batch size: 32
- Optimizer: Adam
- Learning Rate: 1e-3 (pretrain), 1e-5 (fine-tune)
- Loss: Binary Crossentropy

---

## 📊 Evaluation (After Fine-Tuning)

- **Accuracy:** 57%
- **F1-Score (Macro):** 0.57
- **Precision & Recall:** Balanced between both classes

| Class | Precision | Recall | F1-score |
| ----- | --------- | ------ | -------- |
| Bird  | 0.56      | 0.56   | 0.56     |
| Drone | 0.59      | 0.59   | 0.59     |

---

## 📌 Reflections

Transfer learning with fine-tuning proved effective to improve performance from 47% to 57%. MobileNetV2 is suitable for lightweight binary classification with limited data.

---

## 📎 File Structure

├── ml-2-tugas-individu-birds-vs-drone.ipynb
├── Laporan_Tugas_Transfer_Learning_Bird_vs_Drone.pdf
├── README.md
└── images/
├── bird/
└── drone/

---

## 👤 Author

**Muhammad Rafi Aditya**  
442023611057
Informatics Engineering - Universitas Darussalam Gontor
