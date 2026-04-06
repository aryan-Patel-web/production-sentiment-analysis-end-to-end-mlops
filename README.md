<!-- # 🚀 Production Sentiment Analysis — End-to-End MLOps

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![DVC](https://img.shields.io/badge/DVC-Pipeline-purple?logo=dvc)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Container-2496ED?logo=docker)
![AWS EKS](https://img.shields.io/badge/AWS-EKS-FF9900?logo=amazon-aws)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?logo=grafana)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?logo=github-actions)
![License](https://img.shields.io/badge/License-MIT-green)

**A fully production-grade, end-to-end NLP system with a complete MLOps pipeline — from raw data ingestion to live Kubernetes deployment with real-time monitoring.**

[Overview](#-project-overview) · [Architecture](#-system-architecture--flow) · [Folder Structure](#-folder-structure) · [Pipeline](#-dvc-ml-pipeline) · [CI/CD](#-cicd--github-actions) · [EKS Deployment](#-aws-eks-deployment) · [Monitoring](#-monitoring--prometheus--grafana) · [Quick Start](#-quick-start) · [Future Features](#-future-features)

</div>

---

## 📌 Project Overview

This project is a **production-ready sentiment analysis system** that classifies text as **Positive** or **Negative**. It demonstrates a complete, real-world **MLOps lifecycle** — from experiment tracking and data versioning to containerized deployment on a managed Kubernetes cluster with full observability.

The system is built around industry-standard tools used at top ML engineering teams:

| Layer | Tool |
|---|---|
| Data Versioning | DVC + AWS S3 |
| Experiment Tracking | MLflow + DagsHub |
| Serving API | Flask |
| Containerization | Docker + AWS ECR |
| Orchestration | AWS EKS (Kubernetes) |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana |
| Infrastructure | AWS EC2, S3, ECR, EKS |

---

## 🏗 System Architecture & Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        END-TO-END MLOPS FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

  1. DATA LAYER
  ┌──────────────┐    DVC + S3     ┌──────────────┐    preprocessing   ┌────────────────┐
  │  Raw Data    │ ─────────────►  │  Versioned   │ ────────────────►  │  Processed     │
  │  (CSV/text)  │                 │  in AWS S3   │                     │  Features      │
  └──────────────┘                 └──────────────┘                     └────────────────┘
                                                                                │
  2. ML PIPELINE (DVC Stages)                                                   ▼
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │  data_ingestion → data_preprocessing → feature_engineering → model_building     │
  │                                                         → model_evaluation      │
  │                                                         → model_registration    │
  └──────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼ MLflow Tracking (DagsHub)
                               ┌──────────────────────┐
                               │  Experiment Registry  │
                               │  (metrics, params,    │
                               │   artifacts, models)  │
                               └──────────────────────┘
                                          │
  3. CI/CD LAYER (GitHub Actions)         ▼
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │  Push to main ─► Run DVC pipeline ─► Build Docker image ─► Push to ECR     │
  │               └─► Run Tests      └─► Deploy to EKS cluster                 │
  └──────────────────────────────────────────────────────────────────────────────┘
                                          │
  4. SERVING LAYER (AWS EKS)              ▼
  ┌────────────────────────────────────────────────────────────────────────────┐
  │              AWS EKS Cluster (Kubernetes)                                  │
  │                                                                            │
  │   ┌─────────────────────┐        ┌─────────────────────────────────┐      │
  │   │  Flask App Pod #1   │        │  LoadBalancer Service           │      │
  │   │  (replicas: 2)      │◄──────►│  port: 5000 → targetPort: 5000 │◄────►│ User
  │   ├─────────────────────┤        └─────────────────────────────────┘      │
  │   │  Flask App Pod #2   │                                                  │
  │   └─────────────────────┘                                                  │
  │                                                                            │
  │   ┌───────────────────────────────────────────────────────────┐           │
  │   │  ECR Image: 020866158197.dkr.ecr.us-east-1.amazonaws.com │           │
  │   └───────────────────────────────────────────────────────────┘           │
  └────────────────────────────────────────────────────────────────────────────┘
                                          │
  5. MONITORING LAYER                     ▼
  ┌───────────────────────────────────────────────────────────┐
  │   Prometheus  ──scrapes──►  Flask /metrics endpoint       │
  │       │                                                    │
  │       └──────────────►  Grafana Dashboard                  │
  │                         (request rate, latency,            │
  │                          error rate, CPU/memory)           │
  └───────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
production-sentiment-analysis-end-to-end-mlops/
│
├── .dvc/                          ← DVC internal config & cache pointer
│   └── config                     ← Remote S3 storage config
│
├── .github/
│   └── workflows/
│       └── ci-cd.yaml             ← GitHub Actions pipeline (train → build → deploy)
│
├── data/                          ← All data (tracked by DVC, stored in S3)
│   ├── raw/                       ← Original immutable data dump
│   ├── interim/                   ← Cleaned/transformed intermediate data
│   └── processed/                 ← Final ML-ready feature matrices
│
├── docs/                          ← Sphinx project documentation
│
├── flask_app/                     ← Production Flask REST API
│   ├── app.py                     ← Main application (predict endpoint + /metrics)
│   ├── templates/                 ← HTML templates (web UI)
│   └── static/                    ← Static assets (CSS/JS)
│
├── models/                        ← Serialized model artifacts
│   ├── model.pkl                  ← Trained classifier
│   └── vectorizer.pkl             ← TF-IDF vectorizer
│
├── notebooks/                     ← EDA & experimentation Jupyter notebooks
│
├── references/                    ← Data dictionaries, manuals
│
├── reports/
│   ├── metrics.json               ← DVC-tracked evaluation metrics
│   ├── experiment_info.json       ← MLflow run ID, model URI
│   └── figures/                   ← Plots and charts
│
├── scripts/                       ← Utility/helper shell scripts
│
├── src/                           ← Core Python package
│   ├── __init__.py
│   ├── data/
│   │   ├── data_ingestion.py      ← Downloads/splits raw data
│   │   └── data_preprocessing.py ← Text cleaning, noise removal
│   ├── features/
│   │   └── feature_engineering.py← TF-IDF vectorization (max_features=53)
│   └── model/
│       ├── model_building.py      ← Trains ML model
│       ├── model_evaluation.py    ← Evaluates & logs metrics to MLflow
│       └── register_model.py      ← Registers best model to MLflow Model Registry
│
├── temp_model/                    ← Temporary model artifacts during pipeline
│
├── tests/                         ← Unit & integration tests
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
│
├── .dockerignore                  ← Files excluded from Docker build context
├── .dvcignore                     ← Files ignored by DVC tracking
├── .gitignore
├── Dockerfile                     ← Multi-stage Docker image for Flask app
├── LICENSE                        ← MIT License
├── Makefile                       ← Developer commands (make train, make test, etc.)
├── README.md
├── deployment.yaml                ← Kubernetes Deployment + LoadBalancer Service manifest
├── dvc.lock                       ← Locked DVC pipeline state (reproducibility)
├── dvc.yaml                       ← DVC pipeline stage definitions
├── params.yaml                    ← Centralized hyperparameter config
├── projectflow.txt                ← Step-by-step project implementation notes
├── requirements.txt               ← Python dependencies
├── setup.py                       ← Makes src/ pip-installable
├── test_environment.py            ← Python environment sanity check
└── tox.ini                        ← Tox testing configuration
```

---

## 🔄 DVC ML Pipeline

The entire ML pipeline is defined in `dvc.yaml` and orchestrated by DVC. Each stage tracks its **dependencies**, **parameters**, and **outputs** — enabling full reproducibility and incremental runs.

```yaml
# dvc.yaml — 6 sequential stages
stages:
  data_ingestion        → pulls & splits raw data (test_size: 0.19)
  data_preprocessing    → cleans text, removes noise
  feature_engineering   → TF-IDF vectorization (max_features: 53)
  model_building        → trains classifier, saves model.pkl
  model_evaluation      → evaluates, logs metrics to MLflow → reports/metrics.json
  model_registration    → registers best model to MLflow Model Registry
```

**Run the full pipeline:**
```bash
dvc repro
```

**Check pipeline status:**
```bash
dvc status
dvc dag          # visualize the DAG
```

**Push data/artifacts to S3:**
```bash
dvc push
```

### `params.yaml` — Centralized Hyperparameters

```yaml
data_ingestion:
  test_size: 0.19

feature_engineering:
  max_features: 53
```

Changing any param and running `dvc repro` will only re-run affected downstream stages.

---

## 📊 Experiment Tracking — MLflow + DagsHub

All experiments are logged to **DagsHub** (MLflow remote) automatically during the `model_evaluation` stage.

Each run records:
- **Parameters**: model hyperparameters, feature config
- **Metrics**: accuracy, precision, recall, F1-score
- **Artifacts**: `model.pkl`, `vectorizer.pkl`, confusion matrix plots
- **Model Registry**: best model promoted to `Production` stage via `register_model.py`

**View experiments:**
```bash
mlflow ui      # or visit your DagsHub MLflow URL
```

---

## 🐳 Docker — Containerization

The Flask app is containerized using Docker. The image is pushed to **AWS ECR**.

**Build the image locally:**
```bash
docker build -t flask-app .
docker run -p 5000:5000 flask-app
```

**Tag & push to ECR:**
```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  020866158197.dkr.ecr.us-east-1.amazonaws.com

docker tag flask-app:latest \
  020866158197.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest

docker push \
  020866158197.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
```

---

## ⚙️ CI/CD — GitHub Actions

The `.github/workflows/ci-cd.yaml` pipeline runs automatically on every push to `main`.

```
┌─ Trigger: git push main ──────────────────────────────────────────────┐
│                                                                        │
│  Step 1: Checkout code                                                 │
│  Step 2: Set up Python 3.10                                            │
│  Step 3: Install dependencies (requirements.txt)                       │
│  Step 4: Configure DVC remote (AWS S3 credentials)                    │
│  Step 5: dvc pull  →  fetch data & cached artifacts                   │
│  Step 6: dvc repro →  run ML pipeline if params/code changed          │
│  Step 7: Run tests  →  pytest tests/                                   │
│  Step 8: Configure AWS credentials (IAM Role / Secrets)               │
│  Step 9: Build Docker image                                            │
│  Step 10: Push image to AWS ECR                                        │
│  Step 11: Update kubeconfig for EKS cluster                           │
│  Step 12: kubectl apply -f deployment.yaml  →  rolling update on EKS  │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

**GitHub Secrets required:**

| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS authentication |
| `AWS_SECRET_ACCESS_KEY` | AWS authentication |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REPOSITORY` | ECR repo name |
| `EKS_CLUSTER_NAME` | Your EKS cluster name |
| `DAGSHUB_TOKEN` | MLflow / DagsHub access |
| `CAPSTONE_TEST` | App secret (injected as K8s secret) |

---

## ☁️ AWS EKS Deployment

The Flask app runs on **Amazon Elastic Kubernetes Service (EKS)** — a fully managed Kubernetes cluster that handles auto-scaling, self-healing, and rolling deployments.

### Kubernetes Manifest — `deployment.yaml`

```yaml
# Deployment — 2 replicas of the Flask container
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-app
spec:
  replicas: 2                      # High availability: 2 pods running at all times
  selector:
    matchLabels:
      app: flask-app
  template:
    spec:
      containers:
        - name: flask-app
          image: 020866158197.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
          ports:
            - containerPort: 5000
          resources:
            requests:              # Guaranteed minimum resources
              memory: "256Mi"
              cpu: "250m"
            limits:                # Hard resource caps
              memory: "512Mi"
              cpu: "1"
          env:
            - name: CAPSTONE_TEST
              valueFrom:
                secretKeyRef:      # Reads from K8s Secret (not hardcoded)
                  name: capstone-secret
                  key: CAPSTONE_TEST
---
# Service — exposes pods via AWS LoadBalancer
apiVersion: v1
kind: Service
metadata:
  name: flask-app-service
spec:
  type: LoadBalancer               # Provisions an AWS ELB automatically
  selector:
    app: flask-app
  ports:
    - port: 5000
      targetPort: 5000
```

### EKS Setup — Step by Step

**Prerequisites:**
```bash
# Install tools
brew install awscli kubectl eksctl helm
aws configure   # set your Access Key, Secret, Region
```

**1. Create the EKS Cluster:**
```bash
eksctl create cluster \
  --name sentiment-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed
```

**2. Connect kubectl to your cluster:**
```bash
aws eks update-kubeconfig \
  --region us-east-1 \
  --name sentiment-cluster
```

**3. Create the Kubernetes Secret:**
```bash
kubectl create secret generic capstone-secret \
  --from-literal=CAPSTONE_TEST=your_secret_value
```

**4. Deploy the application:**
```bash
kubectl apply -f deployment.yaml
```

**5. Get the external LoadBalancer URL:**
```bash
kubectl get service flask-app-service
# Copy the EXTERNAL-IP and open in browser at port 5000
```

**6. Verify pods are running:**
```bash
kubectl get pods
kubectl logs <pod-name>
kubectl describe pod <pod-name>
```

**7. Scale the deployment manually:**
```bash
kubectl scale deployment flask-app --replicas=4
```

**8. Rolling update (after new ECR push):**
```bash
kubectl rollout restart deployment/flask-app
kubectl rollout status deployment/flask-app
```

---

## 📈 Monitoring — Prometheus & Grafana

The system uses **Prometheus** for metrics collection and **Grafana** for real-time visual dashboards — the industry-standard observability stack.

### How it works

```
Flask App (/metrics endpoint)
        │
        │  HTTP scrape every 15s
        ▼
   Prometheus Server
   (stores time-series metrics)
        │
        │  PromQL queries
        ▼
   Grafana Dashboard
   (live graphs, alerts, panels)
```

### Prometheus Setup on EKS

**1. Add the Prometheus Helm chart repo:**
```bash
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts
helm repo update
```

**2. Install Prometheus into your cluster:**
```bash
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

**3. Verify Prometheus is running:**
```bash
kubectl get pods -n monitoring
```

**4. Port-forward to access Prometheus UI locally:**
```bash
kubectl port-forward svc/prometheus-kube-prometheus-prometheus \
  9090:9090 -n monitoring
# Open: http://localhost:9090
```

**5. Prometheus scrape config (added to your Flask app):**

Your Flask app exposes a `/metrics` endpoint using `prometheus_flask_exporter`. Prometheus scrapes this endpoint every 15 seconds to collect metrics.

```python
# In flask_app/app.py
from prometheus_flask_exporter import PrometheusMetrics
metrics = PrometheusMetrics(app)
```

**Key metrics exposed:**
| Metric | Description |
|---|---|
| `flask_http_request_total` | Total HTTP requests by method/endpoint/status |
| `flask_http_request_duration_seconds` | Request latency histogram |
| `flask_http_request_exceptions_total` | Exception count |
| `process_resident_memory_bytes` | Pod memory usage |
| `process_cpu_seconds_total` | Pod CPU usage |

### Grafana Setup

**1. Port-forward Grafana UI:**
```bash
kubectl port-forward svc/prometheus-grafana \
  3000:80 -n monitoring
# Open: http://localhost:3000
# Default login: admin / prom-operator
```

**2. Grafana is pre-configured with Prometheus as a data source** when installed via the kube-prometheus-stack Helm chart.

**3. Import dashboards:**
- Kubernetes cluster overview: **Dashboard ID 6417**
- Flask app metrics: **Dashboard ID 9528**
- Or build a custom dashboard using PromQL queries:

```promql
# Request rate (per second, last 5 minutes)
rate(flask_http_request_total[5m])

# 95th percentile latency
histogram_quantile(0.95,
  rate(flask_http_request_duration_seconds_bucket[5m]))

# Error rate
rate(flask_http_request_total{status=~"5.."}[5m])

# Pod memory usage
container_memory_usage_bytes{namespace="default"}
```

**4. Set up Grafana Alerts:**
- Alert when error rate > 5%
- Alert when p95 latency > 500ms
- Alert when pod memory > 450Mi

---

## 🚀 Quick Start — Run Locally

### 1. Clone & Install

```bash
git clone https://github.com/aryan-Patel-web/production-sentiment-analysis-end-to-end-mlops.git
cd production-sentiment-analysis-end-to-end-mlops

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export DAGSHUB_TOKEN=your_dagshub_token
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### 3. Pull Data from DVC Remote (S3)

```bash
dvc pull
```

### 4. Run the Full ML Pipeline

```bash
dvc repro
```

### 5. Launch the Flask API

```bash
cd flask_app
python app.py
# Open: http://localhost:5000
```

### 6. Run Tests

```bash
pytest tests/ -v
```

### 7. Use Makefile shortcuts

```bash
make data       # run data ingestion
make train      # run full pipeline
make test       # run test suite
```

---

## 🔌 API Reference

### `POST /predict`

Predicts sentiment of input text.

**Request:**
```json
{
  "text": "This product is absolutely amazing!"
}
```

**Response:**
```json
{
  "sentiment": "Positive",
  "confidence": 0.94
}
```

### `GET /health`

Health check endpoint for Kubernetes liveness probe.

```json
{ "status": "ok" }
```

### `GET /metrics`

Prometheus metrics scrape endpoint (plain text Prometheus format).

---

## 🛠 Tech Stack — Detailed

| Category | Technology | Purpose |
|---|---|---|
| Language | Python 3.10 | Core development |
| ML / NLP | scikit-learn, NLTK | Model training, TF-IDF |
| Data Versioning | DVC | Pipeline reproducibility |
| Artifact Storage | AWS S3 | Remote DVC storage |
| Experiment Tracking | MLflow + DagsHub | Metrics, params, model registry |
| API Framework | Flask | REST API serving |
| Containerization | Docker | App packaging |
| Container Registry | AWS ECR | Docker image storage |
| Orchestration | AWS EKS (Kubernetes) | Production deployment |
| CI/CD | GitHub Actions | Automated train-build-deploy |
| Monitoring | Prometheus | Metrics scraping & alerting |
| Visualization | Grafana | Real-time dashboards |
| Infrastructure | AWS EC2, IAM, VPC | Cloud infrastructure |
| Testing | pytest, tox | Unit & integration tests |
| Task Runner | Makefile | Developer workflow shortcuts |

---

## 🔐 Security Best Practices Implemented

- All secrets stored in **GitHub Secrets** — never hardcoded
- App secrets injected at runtime via **Kubernetes Secrets**
- Docker images built with minimal base layers
- ECR image scanning enabled for vulnerability detection
- IAM roles follow least-privilege principle
- DagsHub tokens scoped to read/write only

---

## 🗺 Future Features

| Feature | Description |
|---|---|
| 🔁 **Model Drift Detection** | Integrate Evidently AI to detect input/output distribution shift over time |
| 🤖 **Transformer Models** | Upgrade from TF-IDF + classical ML to BERT / DistilBERT for higher accuracy |
| 📦 **Helm Chart** | Package Kubernetes manifests as a reusable Helm chart for easy re-deployment |
| 🔀 **A/B Testing** | Serve two model versions simultaneously and route traffic by percentage |
| 📊 **Custom Grafana Dashboards** | Pre-built dashboard JSON for one-click import with all key ML metrics |
| ⚡ **Horizontal Pod Autoscaling** | Auto-scale pods based on CPU/memory thresholds using HPA |
| 🛡 **Rate Limiting** | Add API rate limiting via Flask-Limiter to protect the inference endpoint |
| 🧪 **Shadow Mode Testing** | Route live traffic to a new model in parallel without affecting production |
| 📝 **FastAPI Migration** | Migrate Flask to FastAPI for async support and auto-generated OpenAPI docs |
| 🌐 **Multi-Region Deployment** | Deploy EKS clusters in multiple AWS regions for global low-latency serving |
| 🔔 **PagerDuty Alerts** | Route Grafana alerts to PagerDuty/Slack for on-call incident management |
| 📁 **Feature Store** | Integrate Feast for centralized, versioned feature management |

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Aryan Patel**

[![GitHub](https://img.shields.io/badge/GitHub-aryan--Patel--web-black?logo=github)](https://github.com/aryan-Patel-web)

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

*Built with ❤️ as a production-grade MLOps capstone project*

</div> -->
# 🚀 Production Sentiment Analysis — End-to-End MLOps

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![DVC](https://img.shields.io/badge/DVC-3.53.0-945DD6?logo=dvc&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.15.0-0194E2?logo=mlflow&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0.3-000000?logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![AWS EKS](https://img.shields.io/badge/AWS-EKS-FF9900?logo=amazon-aws&logoColor=white)
![Prometheus](https://img.shields.io/badge/Prometheus-Monitoring-E6522C?logo=prometheus&logoColor=white)
![Grafana](https://img.shields.io/badge/Grafana-Dashboard-F46800?logo=grafana&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF?logo=github-actions&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.1-F7931E?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E)

**A fully production-grade, end-to-end NLP system that classifies movie reviews as Positive or Negative — with a complete MLOps lifecycle from raw data ingestion to live Kubernetes deployment, automated CI/CD, and real-time observability.**

[Overview](#-project-overview) • [Architecture](#-system-architecture--flow) • [Folder Structure](#-folder-structure) • [Pipeline](#-dvc-ml-pipeline) • [Model](#-model-details) • [Flask API](#-flask-api--serving) • [Testing](#-testing) • [CI/CD](#-cicd--github-actions) • [EKS](#-aws-eks-deployment) • [Monitoring](#-monitoring--prometheus--grafana) • [Quick Start](#-quick-start) • [Future Features](#-future-features)

</div>

---

## 📌 Project Overview

This project implements a **production-ready IMDB Sentiment Analysis system** built on top of a complete, real-world **MLOps stack**. It is designed to mirror how ML systems are built, tested, versioned, deployed, and monitored at top engineering companies.

The entire pipeline is automated — a single `git push` triggers model training, evaluation, model promotion, Docker packaging, ECR push, and Kubernetes deployment on AWS EKS.

### What makes this production-grade?

| Concern | Solution |
|---|---|
| **Reproducibility** | DVC pipeline with locked stages & S3 remote |
| **Experiment Tracking** | MLflow hosted on DagsHub (params, metrics, artifacts) |
| **Model Governance** | MLflow Model Registry (None → Staging → Production) |
| **Serving** | Flask + Gunicorn REST API |
| **Containerization** | Docker image pushed to AWS ECR |
| **Orchestration** | AWS EKS (Kubernetes) — 2 replicas + LoadBalancer |
| **CI/CD** | GitHub Actions (15-step automated pipeline) |
| **Observability** | Prometheus custom metrics + Grafana dashboards |
| **Testing** | Model load, signature, performance tests + Flask integration tests |
| **Security** | GitHub Secrets + Kubernetes Secrets (zero hardcoded credentials) |

---

## 🏗 System Architecture & Flow

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                    COMPLETE END-TO-END MLOPS ARCHITECTURE                       ║
╚══════════════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 — DATA                                                                 │
│                                                                                  │
│  GitHub (data.csv) ──data_ingestion.py──► data/raw/ (train.csv + test.csv)     │
│                                                   │ DVC-tracked → AWS S3        │
│  data/raw/ ──data_preprocessing.py──► data/interim/ (cleaned CSVs)             │
│  data/interim/ ──feature_engineering.py──► data/processed/ (BOW matrices)      │
│                                         └──► models/vectorizer.pkl              │
└──────────────────────────────────────────────────┬──────────────────────────────┘
                                                   │
┌──────────────────────────────────────────────────▼──────────────────────────────┐
│  LAYER 2 — ML PIPELINE (DVC)                                                    │
│                                                                                  │
│  model_building.py ──► models/model.pkl  (LogisticRegression L1)                │
│  model_evaluation.py ──► MLflow (DagsHub) logs metrics + artifacts             │
│                       ──► reports/metrics.json + experiment_info.json           │
│  register_model.py ──► MLflow Registry  stage: "Staging"                       │
└──────────────────────────────────────────────────┬──────────────────────────────┘
                                                   │
┌──────────────────────────────────────────────────▼──────────────────────────────┐
│  LAYER 3 — CI/CD (GitHub Actions on every git push)                             │
│                                                                                  │
│  ① checkout → ② Python 3.10 → ③ cache pip → ④ install requirements            │
│  ⑤ dvc repro (run pipeline stages if changed)                                   │
│  ⑥ test_model.py (load + signature + performance tests on Staging model)        │
│  ⑦ promote_model.py (Staging → Production, archive old Production)             │
│  ⑧ test_flask_app.py (home page + predict endpoint integration tests)           │
│  ⑨ aws ecr login → ⑩ docker build → ⑪ docker tag + push to ECR               │
│  ⑫ kubectl setup → ⑬ eks update-kubeconfig → ⑭ create K8s secret             │
│  ⑮ kubectl apply -f deployment.yaml (rolling update, zero downtime)            │
└──────────────────────────────────────────────────┬──────────────────────────────┘
                                                   │
┌──────────────────────────────────────────────────▼──────────────────────────────┐
│  LAYER 4 — SERVING (AWS EKS)                                                    │
│                                                                                  │
│         ┌──────────────────────────────────────────────────────┐                │
│         │           flask-app-cluster (EKS)                     │                │
│         │  Pod #1: flask-app (gunicorn :5000)  ◄──────────────────────► User   │
│         │  Pod #2: flask-app (gunicorn :5000)   LoadBalancer (ELB)              │
│         │  K8s Secret: capstone-secret                          │                │
│         │  Image: ECR .dkr.ecr.us-east-1.amazonaws.com/flask-app│               │
│         └──────────────────────────────────────────────────────┘                │
└──────────────────────────────────────────────────┬──────────────────────────────┘
                                                   │
┌──────────────────────────────────────────────────▼──────────────────────────────┐
│  LAYER 5 — OBSERVABILITY                                                        │
│                                                                                  │
│  Flask /metrics ──scrape 15s──► Prometheus ──PromQL──► Grafana Dashboards      │
│  (app_request_count, app_request_latency_seconds, model_prediction_count)       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
production-sentiment-analysis-end-to-end-mlops/
│
├── .dvc/                              ← DVC internal metadata
│   ├── .gitignore
│   └── config                         ← S3 remote storage endpoint config
│
├── .dvcignore                         ← Files DVC should not track
│
├── .github/
│   └── workflows_stop/
│       └── ci.yaml                    ← Full 15-step GitHub Actions CI/CD pipeline
│
├── .gitignore
│
├── data/                              ← All data (DVC-tracked, stored in AWS S3)
│   ├── raw/
│   │   ├── train.csv                  ← Source data split (test_size=0.19)
│   │   └── test.csv
│   ├── interim/
│   │   ├── train_processed.csv        ← After URL/num/punct/stopword/lemma cleaning
│   │   └── test_processed.csv
│   └── processed/
│       ├── train_bow.csv              ← CountVectorizer BOW matrix (max_features=53)
│       └── test_bow.csv
│
├── docs/                              ← Sphinx documentation project
│   ├── Makefile / make.bat
│   ├── conf.py                        ← Sphinx config
│   ├── index.rst
│   ├── commands.rst
│   └── getting-started.rst
│
├── flask_app/                         ← Production Flask REST API
│   ├── app.py                         ← Routes + text preprocessing + Prometheus metrics
│   ├── preprocessing_utility.py       ← Shared normalize_text() utility
│   ├── load_model_test.py             ← Quick local model loading sanity check
│   ├── requirements.txt               ← Flask-specific dependencies (Flask, gunicorn, prometheus_client...)
│   └── templates/
│       └── index.html                 ← Jinja2 web UI template
│
├── models/                            ← Serialized ML artifacts (DVC-tracked)
│   ├── model.pkl                      ← Trained LogisticRegression classifier
│   └── vectorizer.pkl                 ← Fitted CountVectorizer (BOW, max_features=53)
│
├── notebooks/                         ← Experimentation and EDA
│   ├── IMDB.csv                       ← Raw IMDB dataset
│   ├── data.csv                       ← Working dataset copy
│   ├── exp1.ipynb                     ← Initial EDA and baseline model exploration
│   ├── exp2_bow_vs_tfidf.py           ← Experiment: BOW vs TF-IDF vectorization comparison
│   └── exp3_lor_bow_hp.py             ← Experiment: Logistic Regression hyperparameter tuning
│
├── references/                        ← Data dictionaries, manuals
│
├── reports/
│   ├── metrics.json                   ← DVC-tracked eval metrics (accuracy/precision/recall/Auc)
│   ├── experiment_info.json           ← MLflow run_id + model_path (used by register_model.py)
│   └── figures/                       ← Generated plots and charts
│
├── scripts/
│   └── promote_model.py               ← Archives Production model, promotes Staging → Production
│
├── src/                               ← Core Python package (pip install -e .)
│   ├── __init__.py
│   ├── connections/
│   │   ├── __init__.py
│   │   ├── config.json                ← Connection configuration file
│   │   ├── s3_connection.py           ← AWS S3 fetch operations (alternative data source)
│   │   └── ssms_connection.py         ← SQL Server connection utility
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py          ← load_data, preprocess_data, train_test_split, save_data
│   │   └── data_preprocessing.py     ← preprocess_dataframe: URL/num/lower/punct/stop/lemma
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py    ← apply_bow (CountVectorizer), saves vectorizer.pkl
│   ├── logger/
│   │   └── __init__.py               ← Centralized logging setup (used across all modules)
│   ├── model/
│   │   ├── __init__.py
│   │   ├── model_building.py          ← train_model (LogisticRegression), save_model
│   │   ├── model_evaluation.py       ← evaluate_model, log to MLflow, save_model_info
│   │   ├── predict_model.py           ← Standalone inference utility
│   │   ├── register_model.py          ← register_model → MLflow Registry (Staging)
│   │   └── train_model.py             ← Alternative training entry point
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py               ← Plotting helpers for EDA and evaluation
│
├── tests/
│   ├── test_model.py                  ← 3 tests: load, input/output signature, performance (≥40%)
│   └── test_flask_app.py              ← 2 tests: home page 200 OK, /predict returns sentiment
│
├── Dockerfile                         ← python:3.10-slim + gunicorn (production WSGI)
├── LICENSE                            ← MIT License
├── Makefile                           ← Developer shortcuts: make data, make train, make test
├── README.md
├── deployment.yaml                    ← K8s Deployment (2 replicas) + LoadBalancer Service
├── dvc.lock                           ← Locked pipeline hashes (exact reproducibility)
├── dvc.yaml                           ← DVC 6-stage pipeline definition
├── params.yaml                        ← Centralized hyperparameters (test_size, max_features)
├── projectflow.txt                    ← Step-by-step project implementation notes
├── requirements.txt                   ← Full Python dependency list (pip freeze)
├── setup.py                           ← Makes src/ pip-installable as a package
├── test_environment.py                ← Python environment compatibility check
└── tox.ini                            ← Tox test runner configuration
```

---

## 🔄 DVC ML Pipeline

The ML pipeline is defined in `dvc.yaml` and orchestrated by DVC. Each stage declares its **dependencies**, **parameters**, and **outputs** — enabling incremental execution (only re-runs changed stages) and full reproducibility via `dvc.lock`.

### Pipeline DAG

```
data_ingestion
      │
      ▼
data_preprocessing
      │
      ▼
feature_engineering
      │
      ▼
model_building
      │
      ▼
model_evaluation ──► MLflow (metrics + model artifact)
      │
      ▼
model_registration ──► MLflow Registry (Staging)
```

### Stage Definitions (from `dvc.yaml`)

| Stage | Command | Params | Key Outputs |
|---|---|---|---|
| `data_ingestion` | `python -m src.data.data_ingestion` | `test_size: 0.19` | `data/raw/` |
| `data_preprocessing` | `python -m src.data.data_preprocessing` | — | `data/interim/` |
| `feature_engineering` | `python -m src.features.feature_engineering` | `max_features: 53` | `data/processed/`, `models/vectorizer.pkl` |
| `model_building` | `python -m src.model.model_building` | — | `models/model.pkl` |
| `model_evaluation` | `python -m src.model.model_evaluation` | — | `reports/metrics.json`, `reports/experiment_info.json` |
| `model_registration` | `python -m src.model.register_model` | — | MLflow Model Registry entry |

### `params.yaml`

```yaml
data_ingestion:
  test_size: 0.19          # 19% of data held out for testing

feature_engineering:
  max_features: 53         # Vocabulary size for CountVectorizer (BOW)
```

### DVC Commands

```bash
dvc repro          # Run or resume pipeline (skips unchanged stages)
dvc dag            # Visualize the pipeline as a DAG
dvc status         # Check which stages are stale
dvc params diff    # See parameter changes since last run
dvc metrics show   # Print evaluation metrics from reports/metrics.json
dvc push           # Upload data/artifacts to AWS S3
dvc pull           # Download data/artifacts from AWS S3
```

---

## 🧠 Model Details

### Algorithm — Logistic Regression

```python
# src/model/model_building.py
clf = LogisticRegression(C=1, solver='liblinear', penalty='l1')
```

L1 regularization (Lasso) was chosen after hyperparameter tuning (`exp3_lor_bow_hp.py`). It produces sparse weights, which is ideal for high-dimensional BOW feature spaces.

### Feature Engineering — Bag of Words

```python
# src/features/feature_engineering.py
vectorizer = CountVectorizer(max_features=53)
X_train_bow = vectorizer.fit_transform(X_train)  # fit on train only — no leakage
X_test_bow  = vectorizer.transform(X_test)
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))
```

> **Why BOW over TF-IDF?** Direct comparison in `notebooks/exp2_bow_vs_tfidf.py` showed BOW performed better for this dataset size and feature count.

### Text Preprocessing Pipeline

The same pipeline runs during training (`data_preprocessing.py`) and inference (`flask_app/app.py`) — preventing training-serving skew:

```
Raw text
   │ 1. Remove URLs        (https://, www.)
   │ 2. Remove digits
   │ 3. Lowercase
   │ 4. Remove punctuation (string.punctuation + re.sub)
   │ 5. Remove stop words  (NLTK English stopwords)
   │ 6. Lemmatization      (WordNetLemmatizer)
   ▼
Cleaned text → CountVectorizer.transform() → Feature vector → LogisticRegression.predict()
```

### Evaluation Metrics

```python
# src/model/model_evaluation.py
metrics = {
    'accuracy':  accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall':    recall_score(y_test, y_pred),
    'auc':       roc_auc_score(y_test, y_pred_proba)   # uses predict_proba[:, 1]
}
# All 4 metrics logged to MLflow AND saved to reports/metrics.json
```

---

## 📊 Experiment Tracking — MLflow + DagsHub

All experiments tracked on **DagsHub's hosted MLflow** server. Authentication uses the `CAPSTONE_TEST` token as both MLflow username and password.

```python
# Production auth pattern (used in model_evaluation.py, register_model.py, app.py)
dagshub_token = os.getenv("CAPSTONE_TEST")
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
mlflow.set_tracking_uri('https://dagshub.com/<owner>/<repo>.mlflow')
```

### MLflow Experiment: `my-dvc-pipeline`

Each run logs:
- **Parameters**: `C`, `solver`, `penalty`, `max_features`, `test_size`
- **Metrics**: accuracy, precision, recall, AUC
- **Artifacts**: model artifact at `runs:/<run_id>/model`, `reports/metrics.json`
- **Model**: registered as `"my_model"` in Model Registry

### Model Lifecycle

```
model_evaluation.py ──► MLflow run ──► experiment_info.json (stores run_id)
register_model.py   ──► MLflow Registry: stage "None" → "Staging"
promote_model.py    ──► old Production → "Archived", Staging → "Production"
flask_app/app.py    ──► loads models:/my_model/<Production_version> at startup
```

```python
# scripts/promote_model.py
# Archives current production, promotes staging to production
for version in prod_versions:
    client.transition_model_version_stage(name=model_name, version=version.version, stage="Archived")

client.transition_model_version_stage(name=model_name, version=latest_version_staging, stage="Production")
```

---

## 🐳 Dockerfile — Container Build

```dockerfile
FROM python:3.10-slim           # Minimal base — reduces image size & attack surface

WORKDIR /app

COPY flask_app/ /app/           # Copy entire Flask application
COPY models/vectorizer.pkl /app/models/vectorizer.pkl  # BOW vectorizer (bundled in image)

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet  # Pre-download at build time

EXPOSE 5000

# PRODUCTION: Gunicorn WSGI server with 120s timeout (for MLflow model loading)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

**Key decisions:**
- `python:3.10-slim` — minimal base image
- Gunicorn `--timeout 120` — handles slow MLflow model download on first request
- NLTK data pre-downloaded at image build time (not at container start)
- `model.pkl` NOT bundled — loaded from MLflow Registry at startup (always gets Production version)
- `vectorizer.pkl` IS bundled — static artifact that doesn't change between runs

---

## 🌐 Flask API — Serving

### Routes

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Web UI — renders `index.html` with input form |
| `POST` | `/predict` | Preprocesses text → BOW → model.predict() → renders result |
| `GET` | `/metrics` | Prometheus scrape endpoint (custom registry, plain text) |

### `/predict` Full Flow

```python
@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    text = request.form["text"]
    text = normalize_text(text)           # same 6-step cleaning as training
    features = vectorizer.transform([text])
    features_df = pd.DataFrame(features.toarray(),
                               columns=[str(i) for i in range(features.shape[1])])
    result = model.predict(features_df)   # mlflow.pyfunc model needs DataFrame
    prediction = result[0]

    PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
    REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

    return render_template("index.html", result=prediction)
```

### Prometheus Instrumentation (Custom Registry)

```python
registry = CollectorRegistry()   # Custom registry — isolates app metrics

REQUEST_COUNT    = Counter("app_request_count",
                           "Total requests", ["method", "endpoint"], registry=registry)
REQUEST_LATENCY  = Histogram("app_request_latency_seconds",
                             "Request latency", ["endpoint"], registry=registry)
PREDICTION_COUNT = Counter("model_prediction_count",
                           "Predictions per class", ["prediction"], registry=registry)

@app.route("/metrics")
def metrics():
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}
```

---

## 🧪 Testing

### `tests/test_model.py` — Model Validation (runs against **Staging** model)

```python
class TestModelLoading(unittest.TestCase):

    def test_model_loaded_properly(self):
        # model object should not be None
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # verify input shape matches vectorizer vocabulary
        self.assertEqual(input_df.shape[1], len(vectorizer.get_feature_names_out()))
        # verify output is 1D (binary classification)
        self.assertEqual(len(prediction.shape), 1)

    def test_model_performance(self):
        # all metrics must exceed 40% on holdout test set
        self.assertGreaterEqual(accuracy_new,  0.40)
        self.assertGreaterEqual(precision_new, 0.40)
        self.assertGreaterEqual(recall_new,    0.40)
        self.assertGreaterEqual(f1_new,        0.40)
```

Tests run against the **Staging** model before promotion — bad models are caught before reaching Production.

### `tests/test_flask_app.py` — API Integration Tests

```python
class FlaskAppTests(unittest.TestCase):

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_page(self):
        response = self.client.post('/predict', data=dict(text="I love this!"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(b'Positive' in response.data or b'Negative' in response.data)
```

```bash
# Run all tests
python -m unittest tests/test_model.py -v
python -m unittest tests/test_flask_app.py -v
```

---

## ⚙️ CI/CD — GitHub Actions

**File:** `.github/workflows_stop/ci.yaml`  
**Trigger:** Every `git push`

```
git push
  │
  ├─① checkout code (actions/checkout@v3)
  ├─② setup Python 3.10 (actions/setup-python@v2)
  ├─③ cache pip deps (keyed on requirements.txt hash)
  ├─④ pip install -r requirements.txt
  │
  ├─⑤ dvc repro  [CAPSTONE_TEST]
  │     └─ Runs all 6 pipeline stages, skips unchanged ones
  │
  ├─⑥ python -m unittest tests/test_model.py  [CAPSTONE_TEST]
  │     └─ Load + signature + performance tests on Staging model
  │
  ├─⑦ python scripts/promote_model.py  [CAPSTONE_TEST]  (if tests pass)
  │     └─ Archives Production, promotes Staging → Production
  │
  ├─⑧ python -m unittest tests/test_flask_app.py  [CAPSTONE_TEST]  (if success)
  │     └─ Home page + /predict endpoint integration tests
  │
  ├─⑨ aws configure + ECR login  (if success)
  ├─⑩ docker build -t $ECR_REPOSITORY:latest .
  ├─⑪ docker tag + docker push → AWS ECR
  │
  ├─⑫ azure/setup-kubectl@v3 (install kubectl)
  ├─⑬ aws eks update-kubeconfig --name flask-app-cluster --region us-east-1
  ├─⑭ kubectl create secret generic capstone-secret (--dry-run=client | apply)
  └─⑮ kubectl apply -f deployment.yaml  →  rolling update on EKS
```

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `CAPSTONE_TEST` | DagsHub token (MLflow username + password) |
| `AWS_ACCESS_KEY_ID` | IAM access key |
| `AWS_SECRET_ACCESS_KEY` | IAM secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `AWS_ACCOUNT_ID` | 12-digit AWS account number |
| `ECR_REPOSITORY` | ECR repo name (e.g. `flask-app`) |

---

## ☁️ AWS EKS Deployment

### Kubernetes Manifest — `deployment.yaml`

```yaml
# Deployment — 2 replicas for high availability
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: flask-app
  template:
    spec:
      containers:
        - name: flask-app
          image: 020866158197.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
          ports:
            - containerPort: 5000
          resources:
            requests:
              memory: "256Mi"   # guaranteed minimum for scheduling
              cpu: "250m"
            limits:
              memory: "512Mi"   # hard cap — pod OOMKilled if exceeded
              cpu: "1"
          env:
            - name: CAPSTONE_TEST
              valueFrom:
                secretKeyRef:   # injected from K8s Secret — never hardcoded
                  name: capstone-secret
                  key: CAPSTONE_TEST
---
# Service — AWS ELB via LoadBalancer type
apiVersion: v1
kind: Service
metadata:
  name: flask-app-service
spec:
  type: LoadBalancer
  selector:
    app: flask-app
  ports:
    - port: 5000
      targetPort: 5000
```

### EKS Setup — Full Step-by-Step

**1. Install tools:**
```bash
brew install awscli kubectl eksctl
aws configure   # Access Key, Secret Key, Region: us-east-1, output: json
```

**2. Create the EKS cluster:**
```bash
eksctl create cluster \
  --name flask-app-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 4 \
  --managed
# Takes ~15 minutes. eksctl automatically updates ~/.kube/config
```

**3. Verify connection:**
```bash
kubectl get nodes
kubectl get all
```

**4. Create Kubernetes Secret:**
```bash
kubectl create secret generic capstone-secret \
  --from-literal=CAPSTONE_TEST=<your_dagshub_token>
kubectl get secret capstone-secret   # verify
```

**5. Deploy the application:**
```bash
kubectl apply -f deployment.yaml
```

**6. Get the LoadBalancer URL:**
```bash
kubectl get service flask-app-service
# Wait ~2 min for EXTERNAL-IP
# Access: http://<EXTERNAL-IP>:5000
```

**7. Day-2 operations:**
```bash
# View pod status and logs
kubectl get pods
kubectl logs <pod-name> -f
kubectl describe pod <pod-name>

# Shell into a running pod
kubectl exec -it <pod-name> -- /bin/sh

# Scale replicas
kubectl scale deployment flask-app --replicas=4

# Rolling restart (picks up latest ECR image)
kubectl rollout restart deployment/flask-app
kubectl rollout status deployment/flask-app

# Rollback to previous version
kubectl rollout undo deployment/flask-app

# View resource usage
kubectl top pods
kubectl top nodes
```

**8. Clean up (avoid AWS costs):**
```bash
eksctl delete cluster --name flask-app-cluster --region us-east-1
```

---

## 📈 Monitoring — Prometheus & Grafana

### Architecture

```
Flask App
  └── GET /metrics  (plain text Prometheus exposition format)
          │
          │  scrape every 15s
          ▼
    Prometheus Server
    (time-series TSDB, retention 15d by default)
          │
          │  PromQL
          ▼
    Grafana Dashboards
    (panels: request rate, latency p95, prediction counts, error rate)
```

### Custom Metrics from the Flask App

| Metric | Type | Labels | What it tracks |
|---|---|---|---|
| `app_request_count_total` | Counter | `method`, `endpoint` | Total HTTP requests by route |
| `app_request_latency_seconds` | Histogram | `endpoint` | Request latency distribution |
| `model_prediction_count_total` | Counter | `prediction` | Positive (1) vs Negative (0) count |

Sample `/metrics` output:
```
app_request_count_total{endpoint="/",method="GET"} 42.0
app_request_count_total{endpoint="/predict",method="POST"} 17.0
app_request_latency_seconds_bucket{endpoint="/predict",le="0.1"} 14.0
app_request_latency_seconds_bucket{endpoint="/predict",le="0.5"} 17.0
model_prediction_count_total{prediction="1"} 10.0
model_prediction_count_total{prediction="0"} 7.0
```

### Install Prometheus + Grafana on EKS (Helm)

**1. Add Helm repo:**
```bash
helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts
helm repo update
```

**2. Install kube-prometheus-stack (Prometheus + Grafana + AlertManager):**
```bash
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set grafana.adminPassword=admin123
```

**3. Verify all components are running:**
```bash
kubectl get pods -n monitoring
# Should see: prometheus-server, grafana, alertmanager, node-exporter, kube-state-metrics
```

**4. Access Prometheus UI:**
```bash
kubectl port-forward svc/prometheus-kube-prometheus-prometheus \
  9090:9090 -n monitoring
# Open: http://localhost:9090
# Try: app_request_count_total in the expression browser
```

**5. Access Grafana UI:**
```bash
kubectl port-forward svc/prometheus-grafana \
  3000:80 -n monitoring
# Open: http://localhost:3000
# Login: admin / admin123
# Prometheus data source is pre-configured
```

**6. Scrape config for the Flask app:**
```yaml
additionalScrapeConfigs:
  - job_name: 'flask-sentiment-app'
    static_configs:
      - targets: ['flask-app-service.default.svc.cluster.local:5000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Key PromQL Queries for Grafana Panels

```promql
# Overall request rate (req/s)
rate(app_request_count_total[5m])

# Request rate on /predict endpoint only
rate(app_request_count_total{endpoint="/predict"}[5m])

# Median latency (p50)
histogram_quantile(0.50, rate(app_request_latency_seconds_bucket[5m]))

# 95th percentile latency (p95)
histogram_quantile(0.95, rate(app_request_latency_seconds_bucket[5m]))

# Prediction breakdown (positive vs negative)
rate(model_prediction_count_total[5m])

# Positive sentiment ratio (model drift indicator)
rate(model_prediction_count_total{prediction="1"}[5m])
  /
rate(model_prediction_count_total[5m])
```

### Grafana Alerts (Recommended Setup)

| Alert Name | Condition | Severity |
|---|---|---|
| High Latency | p95 latency > 500ms for 2+ min | Warning |
| High Error Rate | HTTP 5xx rate > 5% | Critical |
| No Predictions | `model_prediction_count_total` not increasing for 5 min | Warning |
| Prediction Drift | Positive ratio < 20% or > 80% for 10 min | Warning |
| Pod Not Running | `kube_pod_status_phase{phase="Running"} < 2` | Critical |

---

## 🚀 Quick Start — Run Locally

**Prerequisites:** Python 3.10, Git, DVC, Docker, AWS CLI configured, DagsHub account

### 1. Clone & Install

```bash
git clone https://github.com/aryan-Patel-web/production-sentiment-analysis-end-to-end-mlops.git
cd production-sentiment-analysis-end-to-end-mlops

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -e .                # Installs src/ as a Python package
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export CAPSTONE_TEST=your_dagshub_token
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

### 3. Pull Data from S3

```bash
dvc pull
```

### 4. Run the Full ML Pipeline

```bash
dvc repro
```

### 5. View Metrics

```bash
dvc metrics show
cat reports/metrics.json
```

### 6. Launch Flask App

```bash
cd flask_app
python app.py
# Open: http://localhost:5000
# Metrics: http://localhost:5000/metrics
```

### 7. Run Tests

```bash
python -m unittest tests/test_model.py -v
python -m unittest tests/test_flask_app.py -v
```

### 8. Makefile Shortcuts

```bash
make data    # Run data ingestion
make train   # Run full DVC pipeline
make test    # Run test suite
```

---

## 🛠 Tech Stack — Complete Reference

| Category | Technology | Version | Role |
|---|---|---|---|
| Language | Python | 3.10 | Core development |
| ML | scikit-learn | 1.5.1 | LogisticRegression, CountVectorizer |
| NLP | NLTK | 3.8.1 | WordNetLemmatizer, stopwords |
| Data | pandas, numpy | 2.2.2, 1.26.4 | Data manipulation |
| Data Versioning | DVC | 3.53.0 | Pipeline reproducibility & S3 remote |
| Artifact Storage | AWS S3 | — | DVC remote backend |
| Experiment Tracking | MLflow | 2.15.0 | Metrics, params, model registry |
| Remote MLflow | DagsHub | 0.3.34 | Hosted MLflow server + Git integration |
| API Framework | Flask | 3.0.3 | REST API |
| WSGI Server | Gunicorn | — | Production HTTP server |
| Monitoring | prometheus_client | — | Counter, Histogram, custom registry |
| Containerization | Docker | — | Image packaging |
| Container Registry | AWS ECR | — | Docker image storage |
| Orchestration | AWS EKS | — | Managed Kubernetes |
| K8s Tools | eksctl, kubectl | latest | Cluster provisioning & management |
| Monitoring Stack | Prometheus | — | Metrics collection via Helm |
| Dashboards | Grafana | — | PromQL visualization & alerting |
| CI/CD | GitHub Actions | — | Automated pipeline |
| Cloud | AWS (S3, ECR, EKS, ELB) | — | Full cloud infrastructure |
| Testing | unittest | stdlib | Model validation + API tests |

---

## 🔐 Security Best Practices

- **Zero hardcoded secrets** — all tokens/keys in GitHub Secrets, injected at runtime
- **Kubernetes Secrets** — `CAPSTONE_TEST` token mounted from `capstone-secret` into pods
- **Minimal base image** — `python:3.10-slim` reduces attack surface
- **IAM least-privilege** — credentials scoped to ECR push + EKS update-kubeconfig only
- **DRY-RUN apply** — K8s secret created with `--dry-run=client -o yaml | kubectl apply` (idempotent)
- **ECR image scanning** — can be enabled for vulnerability detection on push

---

## 🗺 Future Features

| Feature | Description | Priority |
|---|---|---|
| 🔁 **Model Drift Detection** | Evidently AI to detect input/output distribution shift in production | High |
| ⚡ **Horizontal Pod Autoscaler** | Auto-scale pods on CPU/memory thresholds using K8s HPA | High |
| 📦 **Helm Chart** | Package all K8s manifests as a reusable Helm chart | High |
| 📊 **Grafana Dashboard JSON** | Pre-built importable dashboard with all 3 custom metrics | Medium |
| 🤖 **Transformer Upgrade** | Replace BOW + LogReg with DistilBERT for higher accuracy | Medium |
| 🔀 **A/B Model Testing** | Split traffic between two model versions (canary deployment) | Medium |
| 🛡 **API Rate Limiting** | Flask-Limiter to protect /predict from abuse | Medium |
| 📝 **FastAPI Migration** | Async support, auto OpenAPI docs, Pydantic request validation | Medium |
| 🧪 **Shadow Mode** | Run new model in parallel without user impact before promotion | Low |
| 📁 **Feature Store** | Feast integration for centralized, versioned feature management | Low |
| 🌐 **Multi-Region EKS** | Deploy across AWS regions for global low-latency | Low |
| 🔔 **Slack/PagerDuty Alerts** | Route Grafana critical alerts to on-call channels | Low |

---

## 🤝 Contributing

1. Fork this repository
2. Create your feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Aryan Patel**

[![GitHub](https://img.shields.io/badge/GitHub-aryan--Patel--web-181717?logo=github&logoColor=white)](https://github.com/aryan-Patel-web)

---

<div align="center">

**⭐ If this project helped you, please give it a star! ⭐**

*Data Ingestion → BOW Features → Logistic Regression → MLflow Registry → Docker → EKS → Prometheus → Grafana*

</div>