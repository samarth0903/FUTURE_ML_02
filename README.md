# 🎫 Support Ticket Classification & Prioritization System

Machine Learning Task 2 – **Future Interns Internship (2026)**

This project builds an **NLP-based machine learning system** that automatically categorizes customer support tickets and assigns priority levels to help support teams respond faster and manage tickets efficiently.

---

# 📌 Project Overview

In real-world companies, support teams receive **hundreds or thousands of customer tickets every day** through emails, forms, and complaint systems.

Common problems include:

* Tickets are not categorized properly
* Urgent issues get delayed
* Support teams waste time sorting tickets manually

This project solves the problem by using **Natural Language Processing (NLP) and Machine Learning** to:

✔ Automatically classify support tickets into categories
✔ Assign priority levels to tickets
✔ Help support teams respond faster and smarter

---

# 🎯 Objective

The goal of this project is to build a **machine learning system that can:**

1. Read customer support ticket text
2. Automatically classify them into categories such as:

* Hardware
* Access
* HR Support
* Administrative Rights
* Purchase
* Storage
* Miscellaneous
* Internal Project

3. Assign ticket priority levels:

* 🔴 High Priority
* 🟡 Medium Priority
* 🟢 Low Priority

This helps businesses:

* Reduce support backlog
* Improve response time
* Increase customer satisfaction

---

# 🛠️ Technologies Used

## Programming Language

* Python

## Development Environment

* VS Code

## Libraries

* pandas
* numpy
* scikit-learn
* nltk
* matplotlib
* seaborn

These libraries were used for:

* Text preprocessing
* Feature extraction
* Machine learning classification
* Data visualization

---

# 📂 Dataset

Dataset used:

**IT Support Ticket Dataset**

Features included:

| Column      | Description                     |
| ----------- | ------------------------------- |
| Document    | Ticket text / issue description |
| Topic_group | Category of support ticket      |

Dataset size:

* **47,837 tickets**
* **8 categories**

Example:

| Document                                    | Topic_group |
| ------------------------------------------- | ----------- |
| connection with icon icon dear please setup | Hardware    |
| reset passwords for external accounts       | Access      |
| work experience user request                | HR Support  |

---

# ⚙️ Machine Learning Pipeline

The project follows a complete **NLP pipeline**:

### 1️⃣ Text Cleaning

* Convert text to lowercase
* Remove punctuation
* Remove stopwords
* Tokenization using NLTK

Example:

```
"Reset passwords for external accounts"
↓
"reset passwords external accounts"
```

---

### 2️⃣ Feature Extraction

Text is converted into numerical vectors using:

**TF-IDF (Term Frequency – Inverse Document Frequency)**

This helps the machine learning model understand text importance.

Features used:

* **Unigrams + Bigrams**
* **8000 TF-IDF features**

---

### 3️⃣ Model Training

Model used:

**Logistic Regression Classifier**

Reasons for choosing this model:

* Works well for text classification
* Fast training
* Good interpretability

Training split:

```
80% Training Data
20% Testing Data
```

---

# 📊 Model Evaluation

The model was evaluated using standard machine learning metrics.

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~85%  |
| Precision | ~85%  |
| Recall    | ~85%  |

---

# 📈 Confusion Matrix

A confusion matrix was generated to analyze classification performance across categories.

It helps visualize:

* Correct predictions
* Misclassifications
* Class-wise model performance

---

# ⚡ Ticket Priority Assignment

After classification, a **priority system** assigns urgency levels.

Priority logic:

| Category              | Priority |
| --------------------- | -------- |
| Hardware              | High     |
| Access                | High     |
| Purchase              | Medium   |
| Administrative rights | Medium   |
| Others                | Low      |

Example output:

| Ticket            | Category      | Priority |
| ----------------- | ------------- | -------- |
| connection issue  | Hardware      | High     |
| reset password    | Access        | High     |
| mail verification | Miscellaneous | Low      |

---

# 📤 Output

The system generates a file:

```
ticket_predictions.csv
```

Containing:

| Document | Topic_group | clean_text | priority |
| -------- | ----------- | ---------- | -------- |

This allows support teams to **automatically route tickets**.

---

# 📁 Project Structure

```
FUTURE_ML_02
│
├── support_ticket_classifier.py
├── all_tickets_processed_improved_v3.csv
├── ticket_predictions.csv
└── README.md
```

---

# ▶️ How to Run the Project

### 1️⃣ Install dependencies

```
pip install pandas scikit-learn nltk matplotlib seaborn
```

### 2️⃣ Download NLTK resources

```
python -m nltk.downloader punkt
python -m nltk.downloader stopwords
```

### 3️⃣ Run the project

```
python support_ticket_classifier.py
```

Outputs generated:

* Model evaluation metrics
* Confusion matrix
* Ticket priority examples
* Prediction CSV file

---

# 💡 Business Impact

This system helps businesses:

✔ Automatically categorize support tickets
✔ Detect urgent issues faster
✔ Reduce manual sorting work
✔ Improve support team efficiency

Such systems are widely used in:

* SaaS companies
* IT service management platforms
* customer support automation systems

---

# 🚀 Skills Gained

Through this project:

* NLP Text Preprocessing
* Feature Engineering (TF-IDF)
* Multi-class Classification
* Model Evaluation
* Support Operations Optimization

---

# 👨‍💻 Author

**Samarth Gupta**

B.Tech Computer Science & Engineering
VIT-AP University

GitHub:
https://github.com/samarth0903

---

# 🤝 Internship

This project was completed as part of:

**Machine Learning Internship – Future Interns (2026)**

🔗 https://www.linkedin.com/company/future-interns/

---

# 📢 Connect

If you found this project interesting, feel free to ⭐ star the repository or connect with me on GitHub!

---
