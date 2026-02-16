# Socially Aware Spatial MRF Recommender

Clean API version of the notebook project for top-N recommendation using:
- Jaccard similarity
- Bayesian logistic regression
- Social Bayesian Markov model


## Files

- `models.py` - model implementations
- `train_and_save.py` - trains and saves models
- `app.py` - FastAPI inference API
- `requirements.txt` - minimal dependencies
- `Dockerfile`, `docker-compose.yml` - local Docker deployment

## Data

Place dataset files in:
- `epinions_data/epinions.txt`
- `epinions_data/network_trust.txt`

## Free Deployment (Local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_and_save.py 5000
uvicorn app:app --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

## Free Deployment (Docker)

```bash
docker compose up --build
```

## Endpoints

- `GET /health`
- `POST /recommend`
- `POST /predict`

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123","k":10,"model_type":"social_bayesian"}'
```
# Socially Aware Spatial MRF Recommender

Top-N recommendation system using:
- Jaccard similarity baseline
- Bayesian logistic regression
- Social Bayesian Markov model with trust-network signals


## Clean Project Layout

- `epinions_recommendation.ipynb`
- `models.py` - extracted model implementations
- `train_and_save.py` - training + model persistence script
- `app.py` - FastAPI inference API
- `requirements.txt` - minimal dependencies
- `Dockerfile` and `docker-compose.yml` - containerized local deployment

## Free Deployment Options Only

### 1) Local (free, easiest)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_and_save.py 5000
uvicorn app:app --host 0.0.0.0 --port 8000
```

API docs: `http://localhost:8000/docs`

### 2) Docker on your machine (free)

```bash
docker compose up --build
```

### 3) Hugging Face Spaces (free CPU tier)

Use this repo with Docker Space support:
1. Create a new Hugging Face Space (Docker SDK)
2. Push this repo contents
3. Keep app entrypoint as `uvicorn app:app --host 0.0.0.0 --port 8000`

Note: model training can be expensive on free CPUs. Train locally first and include `saved_models/` if needed.

## Data Requirement

Expected files:
- `epinions_data/epinions.txt`
- `epinions_data/network_trust.txt`

## API Endpoints

- `GET /health`
- `POST /recommend`
- `POST /predict`

Example:

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user123","k":10,"model_type":"social_bayesian"}'
```

## Notes

- This repo intentionally stays small and simple.
- Only free deployment paths are documented here.
# Socially-Aware-Spatial-Markov-Random-Field-for-Personalized-Recommendation
Developed top-10 item recommender using Socially-Aware Bayesian Markov Model, integrating social trust, review features, and probabilistic modeling.

## 📺 Project Video

Watch the demo on YouTube: [Epinions Recommendation System Demo](https://youtu.be/fTDacdtgvng?si=ICQoBSLk06JArFKt)

## 👥 Contributors

- **[Sanjana Garimella](https://github.com/sanjana-garimella/)**
- **[Priyanka Nidadavolu](https://github.com/priyanka-nidadavolu)**


## 📌 Project Overview

In this project, we developed a **top-N item recommendation system** using the **Socially-Aware Bayesian Markov Model**, which leverages a **Markov Random Field (MRF)** to refine item rating predictions by incorporating opinions from socially connected neighbors in a **trust network**. The model employs **Bayesian-inspired Logistic Regression** with **35 features**, including:  

- **Review characteristics:** price paid, timestamp, review text length  
- **Social network features:** trust relationships, PageRank scores, centrality measures  
- **Engineered interaction features**

The goal of this project is to analyze how recommendation performance improves when moving from **basic collaborative filtering** to **probabilistic and socially-aware models**, particularly under **extreme sparsity** and **cold-start conditions**.

---

## 🧠 Models Implemented

Three models were implemented and compared using **AUC** to evaluate their ability to rank items that users would rate highly (≥4 stars) within the top-N list:

1. **Jaccard Similarity (Collaborative Filtering baseline)**  
   - User-user and item-item similarity  
   - No content or social features  

2. **Bayesian Logistic Regression**  
   - Uses **3 review features**: price, time, and review text length  
   - Probabilistic model for rating prediction  

3. **Social Bayesian Markov Model (Best Model)**  
   - Combines **Bayesian Logistic Regression with 35 features** + MRF social smoothing  
   - Incorporates **trust network information** to propagate preference signals  

Comparing AUC scores across these models demonstrates how incorporating **social and behavioral features** improves top-N recommendation accuracy over baseline methods.

---

## 📂 Dataset Description

The dataset used in this project is the **Epinions dataset**

This dataset originates from **general consumer reviews collected from an e-commerce platform, Shopping.com**, and was later curated for research and academic use. It contains both **user–item review interactions** and a **user trust network**, making it well-suited for studying recommendation systems with social influence.

### Review Dataset (`epinions.txt`)
- **50,000 reviews**
- **39,719 users**
- **11,197 items**
- **Sparsity:** 99.99%

We deliberately selected the Epinions dataset due to its **extreme sparsity (99.99%)**, which results in limited historical interactions for most users and items.  
This characteristic makes the dataset particularly suitable for studying **cold-start problems** and evaluating the robustness of recommendation models under sparse data conditions.


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
-  Represents which users trust other users’ opinions

This combination enables the exploration of **collaborative**, **content-based**, and **socially aware recommendation models**.

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

- Collaborative filtering fails under **extreme sparsity** and performs close to random.
- The dataset was **intentionally chosen due to its high sparsity (99.99%)**, making it well-suited for studying **cold-start problems** in recommendation systems.
- Probabilistic models provide significant performance gains in sparse settings.
- Social trust information significantly improves recommendation quality by propagating preference signals.
- Combining **content + behavior + social structure** yields the best overall performance.

---

## 🛠️ Technologies & Concepts

**Machine Learning & Recommendation:**
- Recommender Systems
- Logistic Regression
- Bayesian Modeling
- Social Network Analysis
- Markov Random Fields (MRF)
- Feature Engineering
- Sparse Data & Cold-Start Handling

**Deployment & Production:**
- FastAPI (REST API)
- Docker & Docker Compose
- Kubernetes (K8s)
- AWS ECS/Fargate
- Google Cloud Run
- CI/CD (GitHub Actions)
- Nginx (Load Balancing)

---

## 📚 Inspiration & Comparison with Literature

Our project is inspired by the paper: [“Scalable Recommendation with Social Influence and Sequential Modeling” (IJCAI 2017)](https://cseweb.ucsd.edu/~jmcauley/pdfs/ijcai17a.pdf).  

- The paper combines **sequential Markov models** with **Social Bayesian approaches** to leverage both **sequential and social influence**.  
- In contrast, our project replaces the **sequential Markov model** with a **spatial Markov model**, which is better suited for **top-N recommendation tasks**.  
- We also incorporate **social features in a spatial form** to improve personalized recommendations.

**Performance comparison:**

| Model | AUC Score |
|-------|-----------|
| Bayesian (Paper) | 0.53 |
| Sequential Markov + Social Bayesian (Paper) | 0.58 |
| Bayesian (Our Project) | 0.58 |
| Socially-Aware Spatial Markov Random Field (Our Project) | **0.6248** |

**Key Insights:**

- Our model outperforms the paper by **~6% relative improvement** due to the use of **spatial modeling** and enhanced **social feature integration**.  
- Compared to our baseline (Jaccard similarity), the model achieves a **~25% relative improvement**, demonstrating the importance of combining **content, behavior, and social network information** for top-N recommendations.  
- Using a **spatial Markov approach** allows the model to better capture item correlations for top-N recommendation, whereas sequential models are limited to predicting the next item in a sequence.


## 🚀 Deployment

This project now includes a **complete deployment solution** that allows you to deploy the recommendation models anywhere!

### Quick Start (5 minutes)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (use sample for quick testing)
python train_and_save.py 5000

# 3. Start the API
python app.py

# 4. Test the API
curl http://localhost:8000/health
python test_api.py
```

### Deployment Options

| Platform | Setup Time | Best For |
|----------|-----------|----------|
| **Local Docker** | 3 min | Development, testing |
| **Heroku** | 10 min | Quick demos, MVPs |
| **Google Cloud Run** | 10 min | Auto-scaling production |
| **AWS ECS** | 20 min | Enterprise production |
| **Kubernetes** | 30 min | Large-scale deployments |

### API Endpoints

Once deployed, your API provides:

- `POST /recommend` - Get top-k item recommendations
- `POST /predict` - Predict rating probability
- `GET /health` - Health check
- `GET /docs` - Interactive Swagger UI documentation

### Example Usage

```python
import requests

# Get recommendations
response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "user_id": "user123",
        "k": 10,
        "model_type": "social_bayesian"
    }
)
print(response.json())
```

### Deployment Documentation

- 📖 **[Quick Start Guide](QUICKSTART.md)** - Get running in 5 minutes
- 📖 **[Full Deployment Guide](DEPLOYMENT.md)** - Comprehensive guide for all platforms
- 📖 **[Deployment Overview](DEPLOYMENT_README.md)** - Architecture and features
- 💻 **[Client Example](client_example.py)** - Python client code
- 🧪 **[API Tests](test_api.py)** - Test suite for deployed API

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Or build and run manually
docker build -t recommendation-api .
docker run -p 8000:8000 recommendation-api
```

### Cloud Deployment Examples

**Heroku:**
```bash
heroku create my-recommendation-api
git push heroku main
```

**Google Cloud Run:**
```bash
gcloud run deploy --source . --platform managed
```

**AWS ECS:**
```bash
aws cloudformation create-stack \
  --stack-name recommendation-api \
  --template-body file://deploy/aws/cloudformation.yaml
```

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

---

## 📁 Project Structure

```text
epinions-recommendation-system/
│
├── epinions_data/                    # Dataset files
│   ├── epinions.txt
│   └── network_trust.txt
│
├── notebooks/
│   └── epinions_recommendation.ipynb 
│
├── saved_models/                     # Trained model files
│   ├── jaccard_model.pkl
│   ├── bayesian_model.pkl
│   └── social_bayesian_model.pkl
│
├── deploy/                           # Deployment configurations
│   ├── aws/
│   │   ├── ecs-task-definition.json
│   │   └── cloudformation.yaml
│   ├── gcp/
│   │   └── cloudbuild.yaml
│   ├── kubernetes/
│   │   └── deployment.yaml
│   ├── nginx/
│   │   └── recommendation-api.conf
│   └── systemd/
│       └── recommendation-api.service
│
├── .github/workflows/
│   └── deploy.yml                    # CI/CD pipeline
│
├── models.py                         # Model implementations
├── app.py                            # FastAPI application
├── train_and_save.py                 # Model training script
├── client_example.py                 # Python client example
├── test_api.py                       # API test suite
│
├── Dockerfile                        # Docker configuration
├── docker-compose.yml                # Docker Compose setup
├── heroku.yml                        # Heroku deployment
│
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── QUICKSTART.md                     # Quick start guide
├── DEPLOYMENT.md                     # Full deployment guide
└── DEPLOYMENT_README.md              # Deployment overview
```



