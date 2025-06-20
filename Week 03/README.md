# Proyek CNN untuk Klasifikasi Angka Tulisan Tangan (MNIST)

Proyek ini bertujuan membangun model **Convolutional Neural Network (CNN)** menggunakan PyTorch untuk mengenali angka tulisan tangan dari dataset MNIST.

---

## 📁 Struktur Folder

```
COURSE-CNN/
├── assets/
│   ├── t10k-images-idx3-ubyte
│   ├── t10k-labels-idx1-ubyte
│   ├── train-images-idx3-ubyte
│   ├── train-labels-idx1-ubyte
├── .gitignore
├── dataset.py
├── main-dataset.py
├── model.py
├── prediksi_gambar.py
├── train.py
```

---

##  Alur Pengerjaan

### 1. **Baca Dataset — `dataset.py`**

> Membaca data gambar dan label dari file `.idx`, lalu mengubahnya jadi `Tensor` yang bisa dibaca oleh PyTorch.

```python
train_dataset = MNISTDataset(
    images_path='assets/train-images-idx3-ubyte',
    labels_path='assets/train-labels-idx1-ubyte'
)
```

### 2. **Bangun Model CNN — `model.py`**

> Buat arsitektur CNN (lapisan-lapisan otak komputer) agar bisa mengenali pola dari gambar.

```python
class MnistClassifier(nn.Module):
    def __init__(self, num_classes=10):
        ...
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        ...
```

### 3. **Latih Model — `train.py`**

> Training dilakukan dalam beberapa epoch. Di akhir training, model disimpan sebagai `.pth`.

```python
model = MnistClassifier()
...
for epoch in range(args.epochs):
    model.train()
    ...
torch.save(model.state_dict(), 'mnist_model.pth')
```

### 4. **Evaluasi & Visualisasi**

> Melihat seberapa bagus model menjawab soal-soal data tes.

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
...
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
```

### 5. **Uji Gambar Baru — `prediksi_gambar.py`**

> Coba gambar angka buatan sendiri (format JPG/PNG), lalu lihat hasil tebakannya.

```python
image = Image.open('assets/my_digit.jpg')
image = transform(image)
...
output = model(image)
```

---

##  Cara Mendapatkan Visualisasi

###  1. **Confusion Matrix**

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
```

###  2. **Grafik Loss per Epoch**

```python
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, test_losses, label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Train vs Test Loss")
plt.grid(True)
plt.show()
```

###  3. **Prediksi Gambar**

```python
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f'Prediksi: {predicted_label}')
plt.axis('off')
plt.show()
```

---

##  Anggota Tim

| Nama    | Tugas                                                             |
| ------- | ------------------------------------------------------------------|
| Ghulam Mushthofa    | `dataset.py`, `main-dataset.py`, `prediksi_gambar.py` |
| Rafi Aditya | `model.py`                                                    |
| Hasbi Abdullah | `train.py`                                                 |
| Rafii Abdurrahman | Evaluasi & visualisasi                                  |
| Muslih Hadi | Dokumentasi & pengujian                                       |

---

## 📅 Jalankan Training

```bash
python train.py --epochs 10 --batch_size 64 --lr 0.001 --checkpoint mnist_model.pth
```

## 🔍 Jalankan Prediksi Gambar Baru

```bash
python prediksi_gambar.py --image_path assets/7.jpg
```
