"""Configuration for Apache Log DDoS Detection."""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Model
MODEL_PATH = MODEL_DIR / "apache_ddos_model.pt"
TORCHSCRIPT_PATH = MODEL_DIR / "apache_ddos_model.pts"

# Features (16 Apache log features)
FEATURE_NAMES = [
    "request_count",
    "unique_ips", 
    "requests_per_ip",
    "unique_methods",
    "get_ratio",
    "post_ratio",
    "avg_bytes",
    "total_bytes",
    "status_2xx_ratio",
    "status_4xx_ratio",
    "status_5xx_ratio",
    "unique_urls",
    "avg_url_length",
    "request_rate",
    "ip_entropy",
    "url_entropy",
]

INPUT_DIM = len(FEATURE_NAMES)  # 16

# Model Architecture
HIDDEN_LAYERS = [64, 32, 16]
DROPOUT_RATE = 0.3

# Training
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
EARLY_STOP_PATIENCE = 10
TIME_WINDOW = 60  # seconds

