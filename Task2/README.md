# Handwritten Alphabet Character Recognition

A CNN that recognizes handwritten English alphabets (A–Z) from 28×28 grayscale images, trained on the **A-Z Handwritten Alphabets** dataset, with a custom OpenCV pipeline to recognize letters from real photographed images.

> Built as part of the **CodeAlpha Data Science Internship — Task 3**

---

## 📌 Objective

Identify handwritten characters from image data. This project goes beyond digit recognition (MNIST) to classify all 26 uppercase English letters, and extends the trained model to work on actual photographed handwriting — not just clean dataset images.

---

## 📊 Dataset

- **Source:** A-Z Handwritten Alphabets in `.csv` format (Kaggle)
- **File used:** `data.csv`
- **Size:** 372,450 samples × 785 columns (1 label column + 784 pixel columns)
- **Image format:** 28×28 grayscale, flattened and normalized (`/255.`)
- **Classes:** 26 (A–Z)
- **Class distribution:** imbalanced — e.g. `O`: 57,825 samples vs `I`: 1,120 samples (visualized with a bar chart in the notebook)

---

## ⚙️ Approach / Workflow

1. **Load data** — read `data.csv`, convert to a NumPy array (`uint8`) and free the original DataFrame for memory
2. **Reshape & normalize** — pixels reshaped to `(372450, 28, 28)` and scaled to `[0, 1]`
3. **EDA** — class distribution bar chart + a 400-image grid sample to visually sanity-check the data
4. **Train/test split** — 99% / 1% split (large dataset, so 1% test ≈ 3,725 images is still plenty)
5. **Model building** — CNN built with Keras `Sequential`:
   - 4× `Conv2D` layers (128 → 64 → 64 → 64 filters, ReLU, He-uniform init)
   - 2× `MaxPooling2D`
   - `BatchNormalization` (×2)
   - `Dense(100)` → `Dropout(0.1)` → `Dense(64)` → `Dropout(0.125)` → `BatchNormalization`
   - Output: `Dense(26, activation='softmax')`
   - Optimizer: `SGD(lr=0.01, momentum=0.9)`, loss: `sparse_categorical_crossentropy`
   - **471,294 total parameters**
6. **Training** — 5 epochs, 10% validation split
7. **Evaluation** — loss/accuracy curves plotted, final evaluation on held-out test set
8. **Model persistence** — saved and reloaded with `model.save()` / `tf.keras.models.load_model()`
9. **Visual test** — random grid of test predictions, correct (green) vs incorrect (red) labels overlaid
10. **Real-image pipeline** — OpenCV-based function (`alphabet_recognize()`) that:
    - Blurs + thresholds a photographed image
    - Finds character contours and sorts them left-to-right
    - Crops, resizes (18×18), and pads each character to 28×28
    - Runs each through the trained model and prints the predicted letters

---

## 📈 Results

| Metric | Value |
|---|---|
| Final training accuracy (epoch 5) | 99.02% |
| Final validation accuracy (epoch 5) | 99.19% |
| **Test accuracy** | **99.44%** |
| Test loss | 0.02 |

Accuracy climbed steadily across all 5 epochs with no sign of overfitting (train/val curves stayed close), so the model converges fast on this dataset.

---

## 🛠️ Tech Stack

- Python, NumPy, Pandas
- TensorFlow / Keras (CNN, model saving/loading)
- OpenCV (`cv2`) — contour detection & preprocessing for real-image recognition
- Matplotlib — EDA plots, training curves, prediction grids

---

## 📁 Project Structure

```
handwritten-alphabet-recognition/
├── Alphabet_Recognition_Notebook.ipynb   # Main notebook: EDA → CNN training → real-image testing
├── data.csv                              # A-Z Handwritten Alphabets dataset
├── Alphabet_Recognition/                 # Saved TensorFlow model (SavedModel format)
├── 1.jpg ... 6.jpg                       # Sample handwriting images for live testing
└── README.md
```

---

## 🚀 How to Run

```bash
git clone https://github.com/RaviNamdeoo/handwritten-alphabet-recognition.git
cd handwritten-alphabet-recognition
pip install tensorflow numpy pandas matplotlib opencv-python scikit-learn
```

1. Download the **A-Z Handwritten Alphabets in .csv format** dataset from Kaggle and place `data.csv` in the project root.
2. Run `Alphabet_Recognition_Notebook.ipynb` top to bottom in Jupyter Notebook / JupyterLab.
3. To test on your own handwriting, drop a clear photo of letters (white background, dark ink) in the project root and call:

```python
alphabet_recognize('your_image.jpg')
```

---

## 🔍 Sample Usage

```python
import tensorflow as tf
model = tf.keras.models.load_model('Alphabet_Recognition')

alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
[pred] = model.predict(x_test[0].reshape(1, 28, 28, 1))
print(alpha[pred.argmax()])
```

---

## 🔮 Future Improvements

- Train for more epochs with a learning-rate scheduler (5 epochs already hits 99%+, so there's headroom to push further or stop earlier)
- Switch to `Adam` and compare convergence speed against `SGD`
- Extend to lowercase letters and full word/sentence recognition using a CRNN (CNN + RNN + CTC loss)
- Add data augmentation (rotation, slant, noise) to make the real-image pipeline more robust to different handwriting styles
- Fix the incomplete `i+` line in `alphabet_recognize()` — it should be `i += 1`, currently a syntax error if run as-is
- Wrap the model in a Streamlit app with a drawing canvas for live testing

---

## 👤 Author

**Ravi Namdeo**
- GitHub: [@RaviNamdeoo](https://github.com/RaviNamdeoo)
- LinkedIn: [RaviNamdeo](https://linkedin.com/in/ravinamdeo)
