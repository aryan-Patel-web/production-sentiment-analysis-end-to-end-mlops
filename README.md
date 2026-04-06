# 🚀 Production Sentiment Analysis — End-to-End MLOps

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

</div>