# socially-aware-markov-random-field-for-personalized-recommendation
Developed a top-10 item recommendation system for a user using the Socially-Aware Bayesian Markov Model.

# 🛒 Epinions Recommendation System  
**EDA, Probabilistic Models, and Social-Aware Recommendations**

## 📌 Project Overview

This project builds and evaluates a **recommendation system on the Epinions dataset**, a real-world platform containing **user reviews and social trust relationships**. The objective is to analyze how recommendation performance improves when moving from **basic collaborative filtering** to **probabilistic and socially informed models**, especially under **extreme data sparsity**.

Three recommendation models are implemented and compared using **AUC**:
1. Jaccard Similarity (Collaborative Filtering baseline)
2. Bayesian Logistic Regression
3. Social Bayesian Markov Model (Trust-aware)

---

## 📂 Dataset Description

The dataset used in this project is the **Epinions dataset**

This dataset originates from **general consumer reviews collected from an e-commerce platform, Shopping.com**, and was later curated for research and academic use. It contains both **user–item review interactions** and a **user trust network**, making it well-suited for studying recommendation systems with social influence.

### Review Dataset (`epinions.txt`)
- **50,000 reviews**
- **39,719 users**
- **11,197 items**
- **Sparsity:** 99.99%

Each review includes:
- User ID  
- Item ID  
- Rating (1–5 stars)  
- Price paid  
- Timestamp  
- Review text  

### Trust Network
- **84,516 users**
- **1,159,338 directed trust edges**
- Mean in/out degree: **13.72**

---

## 📊 Exploratory Data Analysis (EDA)

Key observations:
- Ratings are positively skewed (mean: **3.61**, median: **4.0**)
- **65%** of ratings are ≥ 4
- **85%** of users have only one review
- Item popularity follows a long-tail distribution
- Trust network contains influential hub users

---

## 🎯 Problem Formulation

- **Task:** Binary rating prediction  
  - Positive class: rating ≥ 4  
  - Negative class: rating < 4  
- **Evaluation Metric:** AUC
- **Train/Test Split:** 80% / 20%
- **Modeling Sample Size:** 10,000 reviews

---

## 🧠 Models Implemented

### 1️⃣ Jaccard Similarity Model (Baseline)

- Memory-based collaborative filtering
- Jaccard similarity between user–item interaction sets
- No content or popularity features

**AUC:** 0.4997

---

### 2️⃣ Bayesian Model (Logistic Regression)

- Probabilistic binary classifier
- Features:
  - Price paid
  - Normalized timestamp
  - Review word count

**AUC:** 0.5830

---

### 3️⃣ Social Bayesian Markov Model (Best Model)

- Bayesian model enhanced with **social trust features**
- Includes:
  - In-degree / out-degree
  - PageRank
  - Neighbor rating patterns
- Combines Logistic Regression with MRF-style smoothing

**AUC:** **0.6248**

---

## 📈 Model Performance Comparison

| Model                   | AUC Score |
|------------------------|-----------|
| Jaccard Similarity     | 0.4997    |
| Bayesian               | 0.5830    |
| Social Bayesian Markov | **0.6248** |

---

## 🧪 Evaluation Metrics

- AUC (primary)
- Precision, Recall, F1-score
- Confusion Matrix

---

## 🔍 Key Insights

- Collaborative filtering fails under extreme sparsity
- Probabilistic models provide large gains
- Social trust significantly improves recommendations
- Combining **content + behavior + social structure** yields best performance

---

## 🛠️ Technologies & Concepts

- Recommender Systems
- Logistic Regression
- Bayesian Modeling
- Social Network Analysis
- Markov Random Fields
- Feature Engineering
- Sparse Data Handling

---

## 🚀 Future Work

- Matrix Factorization (ALS, SVD++)
- Graph Neural Networks (GNNs)
- Temporal modeling
- LLM-based review embeddings
- Cold-start handling

---

## 📁 Project Structure

```text
epinions-recommendation-system/
│
├── data/
│   ├── epinions.txt
│   ├── network_trust.txt
│   └── network_trustedby.txt
│
├── notebooks/
│   └── epinions_recommendation.ipynb
│
├── models/
│   ├── jaccard_model.py
│   ├── bayesian_model.py
│   └── social_bayesian_markov.py
│
├── requirements.txt
└── README.md

