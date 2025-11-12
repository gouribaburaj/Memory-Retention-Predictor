# 🧠 Memory Retention Predictor Using Machine Learning

## 📘 Overview
The **Memory Retention Predictor** is a machine learning-based system designed to estimate how much a learner will remember from their previous study session based on behavioral and environmental features.  
It predicts memory retention levels — **High**, **Medium**, or **Low** — using features such as:
- Sleep hours
- Time since study
- Topic difficulty
- Number of distractions

This project bridges **cognitive science** and **machine learning**, helping learners understand how study habits affect their memory performance.

---

## 🎯 Objective
To develop an intelligent system that predicts a learner’s memory retention level and helps students, teachers, and educational systems make **data-driven learning decisions**.

---

## 🧩 Methodology

### 1. Data Collection  
A **custom dataset** of 50 samples was created with realistic values for:
- Sleep Hours  
- Time Since Study  
- Topic Difficulty  
- Distractions  
The target variable was **Memory Retention (High / Medium / Low)**.

### 2. Preprocessing  
- Label Encoding for categorical outputs  
- Normalization for distance-based algorithms  
- Data split: **80% training**, **20% testing**

### 3. Algorithms Used  
| Algorithm | Type | Description |
|------------|------|-------------|
| Logistic Regression | Linear | Provides probabilistic outputs and interpretability |
| K-Nearest Neighbors (KNN) | Distance-based | Classifies by similarity to known samples |
| Naive Bayes | Probabilistic | Fast, assumes feature independence |
| Decision Tree | Tree-based | Provides interpretability and handles non-linear data |

### 4. Model Evaluation  
Metrics used: **Accuracy**, **Precision**, **Recall**, **F1-Score**.  
Visualization done using **Matplotlib** and **Seaborn**.

---

## ⚙️ Implementation

### 🐍 Libraries Used
- `pandas` – Data loading and preprocessing  
- `numpy` – Numerical operations  
- `scikit-learn` – Machine learning models and metrics  
- `matplotlib`, `seaborn` – Visualization

### 💻 Platform
Implemented in **Google Colab** using Python.  
**Colab Notebook Link:** [Click Here](https://colab.research.google.com/drive/11ipYdO8enjLQMrjbnnd1EPKk2u_NztHV?usp=sharing)

---

## 📊 Results

| Algorithm | Accuracy (%) | Precision | Recall | F1-Score |
|------------|---------------|------------|----------|-----------|
| Logistic Regression | 82 | 0.81 | 0.80 | 0.80 |
| K-Nearest Neighbors | 78 | 0.77 | 0.76 | 0.76 |
| Naive Bayes | 80 | 0.79 | 0.78 | 0.78 |
| Decision Tree | **85** | **0.84** | **0.83** | **0.83** |

**Best Model:** Decision Tree Classifier 🌳  
The Decision Tree outperformed other algorithms, showing strong interpretability and higher accuracy.

---

## 🧠 Insights
- Sleep, topic difficulty, and distractions significantly influence memory retention.  
- Decision Tree visualizes the influence of each factor clearly.  
- The project demonstrates how **behavioral data** can be analyzed using ML for educational improvement.

---

## 🔮 Future Scope
- Increase dataset size with more diverse features.  
- Include additional parameters like:
  - Study duration
  - Stress level
  - Learning method  
- Explore advanced models like **Random Forest**, **SVM**, and **Neural Networks**.

---

## 👩‍🎓 Author
**Gouri B Raj**  
M.Tech in Artificial Intelligence and Machine Learning  
School of Computing, Amrita Vishwa Vidyapeetham  
Under the guidance of **Prof. Dr. Swaminathan J**  
📅 *November 2025*

---

## 📚 References
1. James, G., Witten, D., Hastie, T., & Tibshirani, R. *An Introduction to Statistical Learning*. Springer, 2013.  
2. Han, J., Kamber, M., & Pei, J. *Data Mining: Concepts and Techniques*. Morgan Kaufmann, 2012.  
3. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR, 2011.  
4. McKinney, W. *Python for Data Analysis*. O’Reilly, 2017.  
5. Bishop, C. M. *Pattern Recognition and Machine Learning*. Springer, 2006.  
6. Kuhn, M., & Johnson, K. *Applied Predictive Modeling*. Springer, 2013.

---

## 🏁 Conclusion
This project proves that **machine learning can model cognitive behaviors** like memory retention effectively.  
By understanding how habits affect memory, learners can adopt smarter study practices and educators can design more effective teaching strategies.

---
