# 🩺 ScriptoSolution

[🔗 GitHub Repository](https://github.com/anuragpathak27/ScriptoSolution)

ScriptoSolution is a Doctor Recommendation System that analyzes handwritten prescriptions to recommend suitable medical specialists in the NCR region. It uses a Flask-based interface and machine learning techniques (CNN, ANN, Naive Bayes) for prescription text extraction, medicine classification, disease detection, and specialist matching.

---

## 🚀 Features

- 📷 **Prescription Analysis**: Upload a scanned prescription and extract key information like medicines and symptoms.
- 🧠 **ML-Based Prediction**:
  - Detects possible diseases using medicine data.
  - Recommends the best specialist from the NCR region for accurate medical care.
- 🌐 **User Interface**: Built using Flask for easy interaction and real-time results.
- 📊 **Multi-Model Support**: Utilizes CNN, ANN, and Naive Bayes classifiers for reliable performance.

---

## 🧠 Technologies Used

- **Frontend**: HTML, CSS (Flask Templates)
- **Backend**: Python, Flask
- **Machine Learning**: 
  - CNN (for image/text classification)
  - ANN (for pattern learning and disease prediction)
  - Naive Bayes (for probabilistic classification)
- **Libraries**: Scikit-learn, Pandas, NumPy, OpenCV, PyTesseract (for OCR)

---

## 📂 Datasets Used

1. **Prescription–Medicine Dataset**  [🔗 *(Kaggle)*](https://www.kaggle.com/datasets/mehaksingal/illegible-medical-prescription-images-dataset): Hand-curated dataset containing scanned prescriptions and corresponding medicine lists.
2. **Disease–Medicine Dataset** *(custom)*: Maps various medicines to likely diseases, based on doctor consultations.
3. **Specialist–Disease Dataset** *(custom NCR-specific doctors dataset)*: Maps diseases to the most relevant specialists available in the NCR region.

---

## 🛠️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/anuragpathak27/ScriptoSolution.git
cd ScriptoSolution
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask app
```bash
python app.py
```

Visit http://localhost:5000 in your browser.

## 🤝 Contributions
This project uses both public and custom-built datasets. If you'd like to contribute by improving models or expanding the dataset coverage outside NCR, feel free to open a pull request.
