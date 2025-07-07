# CodSoft_Task4

# 📩 SMS Spam Detection using Machine Learning

This project is part of my **CodSoft ML Internship**. The aim is to build a machine learning model that can automatically classify SMS messages as either **spam** or **ham** (legitimate).

---

## 📌 Objective

To develop a text classification model that can detect **unwanted or spam messages** using supervised machine learning techniques and natural language processing (NLP).

---

## 📁 Dataset

- Source: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- Format: `.csv` file with 2 columns:
  - `v1`: Label (`ham` or `spam`)
  - `v2`: Message text
- Total messages: 5,572  
  - Ham: 4,825  
  - Spam: 747

---

## ⚙️ ML Workflow

1. **Data Preprocessing**
   - Removed unnecessary columns
   - Converted labels to binary (0 = ham, 1 = spam)
   - Cleaned text (lowercase, punctuation & stopword removal)
   - Applied **TF-IDF** for text vectorization

2. **Model Training**
   - **Multinomial Naive Bayes**
   - **Logistic Regression**
   - **Support Vector Machine (SVM)**

3. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix
   - Bar chart for model comparison

---

## 📊 Results

| Model                | Accuracy | Spam Recall | Verdict 💬                 |
|---------------------|----------|-------------|----------------------------|
| Naive Bayes          | **98.4%** | ⭐ Very High  | Best overall for spam ✅     |
| Logistic Regression  | 97.3%    | 👍 High       | Balanced & interpretable ✅ |
| SVM (Linear)         | 97.4%    | 👍 High       | Strong classifier 💪         |

🎯 **Naive Bayes** outperformed other models and is ideal for spam detection based on word frequency.

---

## 📌 Tech Stack

- Python 🐍  
- Libraries: `pandas`, `nltk`, `scikit-learn`, `matplotlib`, `seaborn`  
- Jupyter Notebook

---

## 📷 Visuals

- Bar Chart (Model Accuracies)  
- Confusion Matrix Heatmap  
- Line Plot (Precision, Recall, F1-score by class)

---

## 💡 Learnings

- Understood how NLP integrates with ML models  
- Learned how **TF-IDF** works for spam filtering  
- Saw the power of **Naive Bayes** in text classification

---

## 🚀 Future Work

- Try deep learning (LSTM or BERT) for more complex texts  
- Deploy model using Streamlit or Flask  
- Add a user-friendly UI for live SMS classification

---



---

