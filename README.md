# 🧬 Ovarian Cancer Prediction using AI & Machine Learning

## 📌 Project Overview
Ovarian cancer is one of the leading causes of cancer-related deaths among women worldwide. Early detection is difficult due to vague symptoms and late-stage diagnosis.

This project aims to develop a Machine Learning model that predicts ovarian cancer using clinical biomarker data. The long-term objective is to extend the system into a multimodal AI framework combining biomarker data and medical imaging for improved prediction accuracy.

---

## 🎯 Problem Statement
India ranks second globally in annual new ovarian cancer cases, contributing nearly 15% of the global burden. With over 324,000 new cases reported worldwide each year, ovarian cancer remains a major public health concern.

Traditional diagnostic methods rely on biomarkers and imaging techniques, but each has limitations when used independently. This project explores how AI/ML can assist in early and reliable detection.

---

## 📊 Dataset Description
The dataset includes:

- Clinical biomarker values (CA125, HE4, CEA, ALT, AST, etc.)
- Patient demographic details (Age, Menopause status)
- TYPE column:
  - 0 → Non-cancer
  - 1 → Cancer

Supplementary datasets were analyzed to understand:
- Raw patient data
- Imputed training data
- Test datasets
- Biomarker reference information

---

## ⚙️ Methodology

1. Data Preprocessing
   - Removed unnecessary columns (SUBJECT_ID)
   - Cleaned special symbols (> , <)
   - Handled missing values

2. Feature Selection
   - Used all relevant biomarker features
   - Target variable: TYPE

3. Train-Test Split
   - 60% Training
   - 40% Testing

4. Model Used
   - Random Forest Classifier
   - Logistic Regression (baseline comparison)

---

## 📈 Results & Performance

Confusion Matrix:

[[ 9 4]
[ 3 27]]


- Correctly detected 27 cancer cases
- Missed 3 cancer cases
- Overall accuracy: ~XX%
- Cancer detection rate (Recall): ~90%

Random Forest outperformed the baseline Logistic Regression model, demonstrating better handling of complex relationships between biomarkers.

---

## 🚀 Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Git & GitHub
- Tailwind CSS (for project website)
- GitHub Pages (deployment)

---

## 🌐 Live Project Website

🔗 https://ananya0386.github.io/OvarianCancer_PBL/

---

## 🔮 Future Scope

- Integration of medical image classification (CNN-based model)
- Multimodal learning combining biomarkers + imaging
- Deployment as a clinical decision-support system
- Improved feature engineering for early-stage detection

---

## 👩‍💻 Author

Ananya Kawatra  
PBL 2026  
Department of Computer Science & Engineering

---

> ⚠️ Disclaimer: This project is for academic and research purposes only. It is not a clinical diagnostic tool.
