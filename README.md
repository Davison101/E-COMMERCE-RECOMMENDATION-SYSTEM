# LightGCN Recommender System

A production-quality product recommendation engine using **Light Graph Convolutional Networks (LightGCN)** for e-commerce platforms.

Built for the **Amazon Reviews dataset** (User → Product → Rating/Purchase), with a modular architecture that works with any user-item interaction data.

## 🎯 Objectives

- Process user-item interaction data
- Train a LightGCN recommendation model
- Generate Top-N personalized recommendations
- Evaluate recommendation quality using:
  - Recall@K
  - Precision@K
  - NDCG@K
  - Hit Rate
- Serve recommendations via REST API
- Modular architecture for future deployment

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12+ |
| ML Framework | PyTorch |
| Graph Learning | PyTorch (sparse operations) |
| Backend API | FastAPI + Uvicorn |
| Database | SQLite (PostgreSQL-ready via SQLAlchemy) |
| Data Processing | Pandas, NumPy, SciPy |
| Visualization | Matplotlib |
| Utilities | scikit-learn, tqdm, PyYAML |

## 📁 Project Structure

```
lightgcn-recommender/
│
├── api/                    # FastAPI endpoints (Step 10)
│   ├── main.py             # App factory, lifespan, routes
│   └── schemas.py          # Pydantic request/response models
│
├── config/
│   └── settings.yaml       # Master YAML configuration
│
├── data/
│   ├── raw/                # Raw datasets (place Amazon Reviews CSV here)
│   └── processed/          # Processed splits, mappings, graph
│
├── preprocessing/          # Module 1 — Data preparation
│   ├── load_data.py        # Load + validate raw data
│   ├── clean_data.py       # Deduplicate, handle missing, validate
│   ├── encode.py           # Bidirectional ID mapping (fit/transform/save)
│   └── split_data.py       # Train/val/test split (random/temporal/LOO)
│
├── graph/                  # Module 2 — Graph construction
│   ├── build_graph.py      # Bipartite adjacency + symmetric normalization
│
├── models/                 # Module 3 — LightGCN model
│   ├── lightgcn.py         # LightGCN(nn.Module) — no weights, no non-linearities
│
├── training/               # Module 4 — Training pipeline
│   ├── trainer.py          # BPR loss, negative sampling, early stopping
│
├── recommendation/         # Module 5 — Recommendation engine
│   ├── recommend.py        # Top-N scoring, seen-item exclusion, batch support
│
├── evaluation/             # Module 6 — Evaluation metrics
│   ├── metrics.py          # Recall@K, Precision@K, NDCG@K, HitRate@K
│
├── database/               # Database operations (SQLite / PostgreSQL)
├── services/               # Business logic layer
├── utils/
│   ├── config.py           # Singleton Config (YAML + env override)
│   └── logging_config.py   # Rotating file + console logging
│
├── tests/                  # 35 comprehensive tests
│   ├── test_preprocessing.py
│   ├── test_graph.py
│   ├── test_lightgcn.py
│   ├── test_training.py
│   └── test_recommendation.py
│
├── saved_models/           # Model checkpoints
│   └── best_model.pt       # Best model from training
├── logs/                   # Log files
├── requirements.txt
├── .env                    # Environment variable overrides
└── README.md
```

## 🚀 Installation

### Prerequisites

- Python 3.12+
- pip

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Davison101/E-COMMERCE-RECOMMENDATION-SYSTEM.git
   cd lightgcn-recommender
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   venv\Scripts\activate     # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:

   **Option A — Amazon Reviews (recommended):**
   Download from [Amazon Reviews 2023](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) and place the ratings CSV at:
   ```
   data/raw/amazon_reviews/ratings.csv
   ```

   **Option B — MovieLens 100K (quick start):**
   ```bash
   python -c "
   import urllib.request, zipfile, os
   url = 'http://files.grouplens.org/datasets/movielens/ml-100k.zip'
   urllib.request.urlretrieve(url, 'ml-100k.zip')
   with zipfile.ZipFile('ml-100k.zip', 'r') as z:
       z.extractall('data/raw/')
   os.remove('ml-100k.zip')
   "
   ```
   Then update `config/settings.yaml` → `data.raw_data_path: "data/raw/ml-100k/u.data"`,
   set `separator: "\t"`, `header: null`, and remove `column_mapping`.

## 🧪 Running the Tests

```bash
cd lightgcn-recommender
python -m pytest tests/ -v
```

All **35 tests** should pass:

| Test file | Tests | What it covers |
|-----------|-------|----------------|
| `test_preprocessing.py` | 7 | Loading, cleaning, encoding, splitting, full pipeline |
| `test_graph.py` | 1 | Bipartite structure, symmetry, normalization, save/load |
| `test_lightgcn.py` | 1 | Forward pass, gradients, layer embeddings, score matrix |
| `test_training.py` | 1 | 3-epoch integration test with checkpointing |
| `test_recommendation.py` | 27 | Metric functions, Recommender, Evaluator |

## 📊 Pipeline (10 Steps)

```
 1. Dataset         Amazon Reviews (or any user-item data)
       ↓
 2. Load Data      DataLoader.load_data()
       ↓
 3. Clean Data     DataCleaner.clean() — dedup, missing, validate
       ↓
 4. Encode IDs     IDEncoder.fit_transform() — user/item → 0-based indices
       ↓
 5. Split          DataSplitter.split() — train/val/test
       ↓
 6. Build Graph    GraphBuilder — bipartite adjacency + symmetric norm
       ↓
 7. Train Model    Trainer — BPR loss, negative sampling, early stopping
       ↓
 8. Recommend      Recommender — Top-N scoring, seen-item exclusion
       ↓
 9. Evaluate       Evaluator — Recall@K, Precision@K, NDCG@K, HitRate@K
       ↓
10. Serve API      FastAPI — /recommend, /evaluate, /health, /info
```

## 🔌 API Endpoints

Start the API:

```bash
python api/main.py
```

Open **http://127.0.0.1:8000/docs** for the interactive Swagger UI.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Service status + model info |
| `GET` | `/info` | Model configuration & dimensions |
| `GET` | `/recommend/{user_id}` | Top-N for a single user |
| `POST` | `/recommend/batch` | Top-N for multiple users |
| `GET` | `/evaluate` | Recall@K, Precision@K, NDCG@K, HitRate@K |
| `POST` | `/reload` | Hot-reload model checkpoint |

**Example request:**
```bash
curl http://127.0.0.1:8000/recommend/0?top_k=5
# → {"user_id":0,"recommendations":[18,11,19,1,7]}
```

## ⚙️ Configuration

Configuration is managed via `config/settings.yaml`, with optional overrides in `.env`.

Key settings:

```yaml
data:
  raw_data_path: "data/raw/amazon_reviews/ratings.csv"
  expected_columns: ["user_id", "item_id", "rating", "timestamp"]
  column_mapping:                          # Auto-rename Amazon columns
    reviewerID: "user_id"
    asin: "item_id"
    overall: "rating"
    unixReviewTime: "timestamp"
  rating_scale: [1, 5]

model:
  embedding_dim: 64
  n_layers: 3
  learning_rate: 0.001
  epochs: 100

training:
  device: "auto"                           # auto, cpu, cuda
  early_stopping_patience: 10

api:
  host: "0.0.0.0"
  port: 8000
```

## 📝 Module Descriptions

### Module 1 — Preprocessing (`preprocessing/`)
- **DataLoader** — Load CSV/Parquet/JSON, validate schema, generate summary stats
- **DataCleaner** — Remove duplicates, handle missing values, validate IDs/ratings
- **IDEncoder** — Bidirectional mapping (user2id / id2user, item2id / id2item), save/load JSON
- **DataSplitter** — Random, temporal, leave-one-out, and user-based splits

### Module 2 — Graph (`graph/`)
- **GraphBuilder** — Builds bipartite adjacency matrix, applies symmetric normalization (D⁻¹/² @ A @ D⁻¹/²), converts to PyTorch sparse tensors

### Module 3 — Model (`models/`)
- **LightGCN** — Pure embedding-propagation model (no weight matrices, no non-linearities). Configurable layers, alpha mixing, dropout

### Module 4 — Training (`training/`)
- **Trainer** — BPR (Bayesian Personalized Ranking) loss, mini-batch negative sampling, validation loop, early stopping, best-model checkpointing

### Module 5 — Recommendation (`recommendation/`)
- **Recommender** — Scores all candidate items, excludes seen interactions, returns ranked Top-N lists. Supports single, batch, and all-user modes

### Module 6 — Evaluation (`evaluation/`)
- **Evaluator** — Computes Recall@K, Precision@K, NDCG@K, HitRate@K averaged across users. Full-score or batched scoring modes

### API (`api/`)
- **FastAPI** app with Swagger docs, Pydantic schemas, lifespan-managed model loading, health checks

## 📈 Coding Standards

- PEP 8 compliant
- Type hints throughout
- Comprehensive docstrings (NumPy style)
- Logging via rotating file handler + console
- Configuration via YAML + env overrides
- Singleton Config class
- Modular, independently testable modules

## 📚 Dataset

The pipeline is designed for the **Amazon Reviews** dataset with the following schema (auto-mapped):

| Amazon column | Pipeline column | Description |
|---------------|-----------------|-------------|
| `reviewerID` | `user_id` | Unique user identifier |
| `asin` | `item_id` | Unique product identifier |
| `overall` | `rating` | Rating (1–5) |
| `unixReviewTime` | `timestamp` | Interaction timestamp |

The column mapping is fully configurable — any dataset with user, item, rating, and timestamp columns can be used.

The code is designed to easily swap in other datasets (Amazon Reviews, Alibaba, etc.).

## 📄 License

This project is part of a final-year academic project.

## 👥 Author

Final Year Project - LightGCN Recommender System