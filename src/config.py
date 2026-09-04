from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "IMDB Dataset.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MLRUNS_DIR = BASE_DIR / "mlruns"
MLFLOW_DB = BASE_DIR / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"
BEST_MODEL_METADATA_PATH = ARTIFACTS_DIR / "best_model.json"

EXPERIMENT_NAME = "CP4_IMDB_MLP"

MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
MIN_DF = 2
MAX_DF = 0.95
SUBLINEAR_TF = True

TEST_SIZE = 0.20
RANDOM_STATE = 42

EXPERIMENTS = [
    {"name": "MLP-01", "hidden_layer_sizes": (64,), "activation": "relu", "learning_rate_init": 0.001, "batch_size": 128, "max_iter": 20, "alpha": 0.0001},
    {"name": "MLP-02", "hidden_layer_sizes": (64, 64), "activation": "relu", "learning_rate_init": 0.001, "batch_size": 128, "max_iter": 20, "alpha": 0.0001},
    {"name": "MLP-03", "hidden_layer_sizes": (128, 64), "activation": "relu", "learning_rate_init": 0.001, "batch_size": 128, "max_iter": 20, "alpha": 0.0001},
    {"name": "MLP-04", "hidden_layer_sizes": (64, 64), "activation": "tanh", "learning_rate_init": 0.001, "batch_size": 128, "max_iter": 20, "alpha": 0.0001},
    {"name": "MLP-05", "hidden_layer_sizes": (64, 64), "activation": "relu", "learning_rate_init": 0.0003, "batch_size": 128, "max_iter": 25, "alpha": 0.0001},
]
