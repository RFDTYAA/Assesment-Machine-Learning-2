# 🚀 Machine Learning 2: Projects & Explorations

Welcome to my coursework repository for **Machine Learning 2**, completed during the 5th semester at **Universitas Darussalam Gontor**. This collection showcases a series of projects that apply advanced machine learning concepts, from foundational neural networks to more complex architectures like CNNs, RNNs, and Autoencoders.

Each weekly project represents a hands-on implementation of core theoretical concepts, demonstrating a progressive journey through the field of deep learning.

---

##  Weekly Projects Overview

Here is a summary of the projects included in this repository, week by week.

### 🧠 **Week 01: Foundations of Machine Translation (Group Study)**
This initial project was a case study exploring the complexities of machine translation. The main focus was on understanding the nuances of human language—such as context, emotion, and culture—that make direct translation challenging. The key takeaway was that effective translation requires models that can learn from vast amounts of examples to capture patterns and context, rather than just relying on rigid rules and vocabulary lists.

### 🔢 **Week 02: Neural Network Basics - Iris Classification**
This project served as an introduction to building neural networks with **PyTorch**.
* **Objective**: To classify the classic Iris flower dataset.
* **Architecture**: A simple Multi-Layer Perceptron (MLP) was constructed with a Linear layer and a ReLU activation function.
* **Outcome**: The model was successfully trained over 100 epochs, and its performance was compared against a standard Scikit-learn Logistic Regression model, which achieved 100% accuracy on the test set.

### ✍️ **Week 03: CNN for MNIST Handwritten Digit Classification (Group Project)**
This group project involved building a **Convolutional Neural Network (CNN)** from scratch to recognize handwritten digits from the MNIST dataset.
* **Objective**: To achieve high-accuracy classification on the benchmark MNIST dataset.
* **Architecture**: The CNN consisted of convolutional layers for feature extraction, max pooling layers for dimensionality reduction, and fully connected layers for final classification.
* **Outcome**: The model demonstrated strong performance, achieving over **98% accuracy** on the test data, showcasing the power of CNNs for image-based tasks.

### 🦅 **Week 04: Transfer Learning - Bird vs. Drone Classification**
This individual assignment explored the concept of **Transfer Learning** to solve a binary image classification problem with a limited dataset.
* **Objective**: To classify images as either a "bird" or a "drone."
* **Model**: A pre-trained **MobileNetV2** model (trained on ImageNet) was used.
* **Strategy**: The initial model was used for feature extraction, and then several final layers were "fine-tuned" to adapt the model to the specific dataset. This approach improved the model's accuracy from 47% to **57%**.

### 🎬 **Week 05: Sentiment Analysis with RNNs (LSTM)**
This project delved into Natural Language Processing (NLP) by building a model to classify movie reviews.
* **Objective**: To classify sentiment in IMDb movie reviews as either "positive" or "negative."
* **Architecture**: A Recurrent Neural Network (RNN) using a **Long Short-Term Memory (LSTM)** layer was implemented. This architecture is ideal for processing sequential data like text.
* **Process**: The workflow included text tokenization, padding sequences, and training the LSTM model. Although the model showed signs of overfitting due to the limited dataset (100 positive and 100 negative reviews), it served as a practical introduction to building RNNs for text classification.

### 🎨 **Week 06: Image Reconstruction with Autoencoders**
This final exploration focused on representation learning by building a **Convolutional Autoencoder**.
* **Objective**: To compress and reconstruct images from the **Fashion MNIST** dataset.
* **Architecture**: The model featured an **encoder** that down-sampled images into a low-dimensional latent space and a **decoder** that reconstructed the images from this compressed representation.
* **Outcome**: The model successfully learned to reconstruct the general shapes and structures of clothing items, demonstrating its ability to capture essential visual features without supervision. The training loss showed stable convergence over 20 epochs.

---

## 💻 Technology Stack

* **Languages**: Python
* **Core Libraries**: PyTorch, TensorFlow/Keras, Scikit-learn
* **Data Manipulation**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn

---

## 👤 Author

* **Muhammad Rafi Aditya**
* NIM: 442023611057
* **Universitas Darussalam Gontor**

### Acknowledgments
A special thank you to our lecturer, **Dr. Oddy Virgantara Putra, S.Kom., M.T.**, for his guidance and support throughout this course.
