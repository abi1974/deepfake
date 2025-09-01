# 🛡️ Guardian AI: Deepfake Detection Using Deep Learning

Welcome to **Guardian AI**, an advanced deepfake detection system built to combat the rising threat of manipulated digital media. By leveraging **deep learning** architectures, Guardian AI accurately identifies and classifies deepfake images and videos, ensuring digital authenticity and trust.

---

## 🚨 The Problem

Deepfake technology, powered by **Generative Adversarial Networks (GANs)**, can create hyper-realistic fake content that poses a serious risk:

* ⚠️ Misinformation and fake news
* 🕵️ Identity theft
* 💻 Cybersecurity threats

Guardian AI delivers a **robust solution** by using state-of-the-art deep learning models to detect and explain deepfakes in both images and videos.

---

## ✨ Features

* 👥 **Multi-Face Detection** → Detects deepfakes even in images with multiple faces.
* ⚡ **Fast Computation** → Optimized for speed without sacrificing accuracy.
* 🔄 **Dual-Module Architecture** → Separate pipelines for video and image detection.
* 🎞️ **Frame-Based Analysis** → For videos, performs frame-by-frame inspection to uncover manipulations.
* 🎯 **High Accuracy** →

  * **Video Detection:** 87% accuracy on training dataset.
  * **Image Detection:** 95% accuracy on training dataset.
* 🔍 **Explainable Results** → Provides insights into why content was flagged as fake.

---

## 🧠 Core Technology & Methodology

### 🎞️ Video Detection

* **Preprocessing:** Extract & crop faces frame by frame.
* **Feature Extraction:** Uses **ResNeXt50\_32x4d CNN** for high-dimensional feature vectors.
* **Temporal Analysis:** **LSTM RNN** captures temporal inconsistencies unique to deepfakes.
* **Classification:** Outputs "Real" or "Fake" prediction.

### 🖼️ Image Detection

* **Preprocessing:** Images resized & normalized.
* **MesoNet Architecture:** Specialized CNN tuned to detect manipulation cues & residual noise.
* **Classification:** Directly predicts whether image is "Real" or "Fake."

---

## 📊 Comparison with Existing Models

* 🏆 **ResNeXt-LSTM** outperforms existing video detection methods with higher accuracy.
* 🏆 **MesoNet** surpasses baseline models in **F1 score** and **AUC** for image detection.

---

## ⚙️ Implementation Details

* **Backend:** Django framework hosting the models.
* **Frontend:** HTML, CSS, JavaScript for user interaction.
* **Deep Learning:** Trained with **NVIDIA GTX 1650 GPU**.
* **Libraries:** PyTorch, TensorFlow, OpenCV, NumPy, Django.

---

## 🚀 Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/guardian-ai.git
   cd guardian-ai
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your GPU environment (CUDA/cuDNN recommended).

4. Run the Django server:

   ```bash
   python manage.py runserver
   ```

5. Access the web app via:

   ```
   http://127.0.0.1:8000
   ```

---

## 🤝 Contribution

We welcome contributions! 🚀

* 🛠️ Add new features
* 🐞 Report or fix bugs
* 📚 Improve documentation

Check out our **CONTRIBUTING.md** for details.

---

## 👨‍💻 Authors

Made with dedication by:

* Anoop
* Alen
* Ashwin
* Abhiraj

---

## 🔮 Future Enhancements

* 🎤 Real-time deepfake detection via live video streams.
* ☁️ Cloud deployment for scalable usage.
* 📈 Enhanced visualization of detection results.
* 🛡️ Integration with cybersecurity tools for enterprise use.

---

⚡ *Guardian AI — Defending Digital Truth with Deep Learning*
