# LightGCN Recommender System

A production-quality product recommendation engine using **Light Graph Convolutional Networks (LightGCN)** for e-commerce platforms.

## 📋 Project Overview

This project implements a LightGCN-based recommendation engine that learns from historical user-item interaction data to generate Top-N personalized product recommendations. The system is designed as a final-year academic project focused on the recommendation logic itself, with modular architecture suitable for future deployment.

## 🎯 Objectives

- Process user-item interaction data
- Train a LightGCN recommendation model
- Generate Top-N personalized recommendations
- Evaluate recommendation quality using:
  - Recall@K
  - Precision@K
  - NDCG@K
  - Hit Ratio
- Save trained models
- Produce evaluation reports
- Modular architecture for future deployment

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12+ |
| ML Framework | PyTorch |
| Graph Learning | PyTorch Geometric |
| Backend API | FastAPI |
| Database | SQLite (PostgreSQL-ready) |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Utilities | Scikit-Learn, TQDM |

## 📁 Project Structure

```
lightgcn-recommender/
│
├── api/                    # FastAPI endpoints
├── config/                 # Configuration files
│   └── settings.yaml       # Main configuration
├── data/
│   ├── raw/                # Raw datasets
│   ├── processed/          # Processed data
│   └── preprocessing/      # Data preprocessing modules
├── preprocessing/          # Preprocessing scripts (Module 1)
│   ├── load_data.py        # Data loading
│   ├── clean_data.py       # Data cleaning
│   ├── encode.py           # ID encoding
│   └── split_data.py       # Train/val/test splitting
├── graph/                  # Graph construction
├── models/                 # LightGCN model
├── training/               # Training loops
├── recommendation/         # Recommendation generation
├── evaluation/             # Evaluation metrics
├── database/               # Database operations
├── services/               # Business logic
├── utils/                  # Utility modules
│   ├── config.py           # Configuration management
│   └── logging_config.py   # Logging setup
├── notebooks/              # Jupyter notebooks
├── tests/                  # Test suite
├── saved_models/           # Trained model checkpoints
├── logs/                   # Log files
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── main.py                 # Entry point
```

## 🚀 Installation

### Prerequisites

- Python 3.12+
- pip (Python package installer)

### Setup

1. **Clone the repository** (or extract the project files)

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the dataset**:
   - Download MovieLens 100K from: https://grouplens.org/datasets/movielens/
   - Place `u.data` in `data/raw/ml-100k/`

## 📊 Module 1: Data Preprocessing

The preprocessing module handles loading, cleaning, encoding, and splitting interaction data.

### Components

| File | Purpose |
|------|---------|
| `load_data.py` | Load and validate raw datasets |
| `clean_data.py` | Remove duplicates, handle missing values, validate IDs/ratings |
| `encode.py` | Map user/item IDs to continuous indices |
| `split_data.py` | Split data into train/validation/test sets |

### Usage

```python
from preprocessing.load_data import DataLoader
from preprocessing.clean_data import DataCleaner
from preprocessing.encode import IDEncoder
from preprocessing.split_data import DataSplitter

# 1. Load data
loader = DataLoader()
df = loader.load_data("data/raw/ml-100k/u.data")
loader.validate_schema(df)
loader.print_summary(df)

# 2. Clean data
cleaner = DataCleaner()
df = cleaner.clean(df)

# 3. Encode IDs
encoder = IDEncoder()
df = encoder.fit_transform(df)
encoder.save_mappings("data/processed/mappings")

# 4. Split data
splitter = DataSplitter()
train_df, val_df, test_df = splitter.split(df)
splitter.save_splits(train_df, val_df, test_df, "data/processed")
```

### Running Tests

```bash
python tests/test_preprocessing.py
```

## ⚙️ Configuration

Configuration is managed via `config/settings.yaml`. Key settings:

```yaml
data:
  raw_data_path: "data/raw/ml-100k/u.data"
  expected_columns: ["user_id", "item_id", "rating", "timestamp"]
  rating_scale: [1, 5]

preprocessing:
  min_interactions_per_user: 5
  min_interactions_per_item: 5
  validation_split: 0.1
  test_split: 0.1
  random_seed: 42
```

## 🧪 Testing

Run the test suite:

```bash
python tests/test_preprocessing.py
```

## 📈 Data Flow

```
Dataset → Data Loading → Cleaning → Encoding → Train/Test Split
→ Graph Construction → LightGCN Training → Recommendation Generation
→ Evaluation → API
```

## 📝 Coding Standards

- PEP 8 compliant
- Type hints throughout
- Comprehensive docstrings
- Logging instead of print statements
- Configuration via YAML files
- Exception handling
- Class-based design
- Modular architecture

## 📚 Dataset

Initially uses **MovieLens 100K** with columns:
- `user_id`
- `item_id`
- `rating`
- `timestamp`

The code is designed to easily swap in other datasets (Amazon Reviews, Alibaba, etc.).

## 📄 License

This project is part of a final-year academic project.

## 👥 Author

Final Year Project - LightGCN Recommender System