# 🎨 Eksplorasi Autoencoder - Fashion MNIST

Proyek ini merupakan bagian dari tugas eksplorasi mandiri pada mata kuliah Deep Learning, dengan tujuan memahami bagaimana Autoencoder bekerja dalam merepresentasikan dan merekonstruksi citra menggunakan pendekatan encoder-decoder.

## 📂 Dataset

Dataset yang digunakan adalah **Fashion MNIST**, terdiri dari 70.000 gambar grayscale 28x28 piksel dari berbagai kategori pakaian.

## 🧠 Arsitektur Model

Model menggunakan **Convolutional Autoencoder** dengan arsitektur berikut:

### Encoder

- Conv2D(32, kernel_size=3, activation='relu') + MaxPooling2D
- Conv2D(16, kernel_size=3, activation='relu') + MaxPooling2D
- Conv2D(8, kernel_size=3, activation='relu') + MaxPooling2D

### Decoder

- Conv2D(8, kernel_size=3, activation='relu') + UpSampling2D
- Conv2D(16, kernel_size=3, activation='relu') + UpSampling2D
- Conv2D(32, kernel_size=3, activation='relu') + UpSampling2D
- Conv2D(1, kernel_size=3, activation='sigmoid')

### Konfigurasi

- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Epoch: 20, Batch size: 256

## 📈 Evaluasi & Visualisasi

- Rekonstruksi gambar input vs output
- Grafik loss selama pelatihan
- Latent space (opsional) menggunakan encoder
- Model menunjukkan performa stabil tanpa overfitting

## 🧪 Eksperimen Tambahan

- Menambahkan dropout (opsional)
- Uji coba ukuran latent space yang berbeda
- Potensi eksplorasi ke model Variational Autoencoder (VAE)

## 📌 Cara Menjalankan

Jalankan notebook berikut:

```bash
pip install tensorflow matplotlib seaborn
Buka notebook eksplorasi-autoencoder.ipynb lalu jalankan semua sel secara berurutan.

\Assesment - Machine Learning 2\Week 06
├── eksploasi-autoencoder.ipynb
├── Laporan Tugas Individu - Eksplorasi Autoencoder.pdf
├── README.md
```

##👤 Author
- Muhammad Rafi Aditya
- 442023611057
- Universitas Darussalam Gontor
