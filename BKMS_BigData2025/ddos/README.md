# Web Attack Detection for Apache Logs

Detects web attacks (SQL Injection, XSS, LFI, SSRF, etc.) from Apache access logs using MLP.

## Training Dataset

**WebAttack-CVSSMetrics** from HuggingFace:
- https://huggingface.co/datasets/chYassine/WebAttack-CVSSMetrics
- 18,842 labeled Apache logs
- 7 attack types + normal traffic

| Attack Type | Count |
|-------------|-------|
| Normal | ~10,000 |
| LFI | 3,088 |
| SSTI | 2,105 |
| SQL Injection | 1,706 |
| XSS | 648 |
| SSRF | 576 |
| File Upload | 437 |
| CSRF | 282 |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r ddos/requirements.txt
```

### 2. Prepare Training Data
```bash
python -m ddos.prepare_data --output dataset/train.csv
```

### 3. Train Model
```bash
python -m ddos.train --data dataset/train.csv --epochs 100
```

### 4. Deploy (Optional)
```bash
cd demo/simple
docker-compose up -d
```

## Features (16 dimensions)

| # | Feature | Description |
|---|---------|-------------|
| 1 | request_count | Total requests in window |
| 2 | unique_ips | Distinct IP addresses |
| 3 | requests_per_ip | Avg requests per IP |
| 4 | unique_methods | Distinct HTTP methods |
| 5 | get_ratio | GET requests ratio |
| 6 | post_ratio | POST requests ratio |
| 7 | avg_bytes | Mean response bytes |
| 8 | total_bytes | Sum response bytes |
| 9 | status_2xx_ratio | Success responses ratio |
| 10 | status_4xx_ratio | Client errors ratio |
| 11 | status_5xx_ratio | Server errors ratio |
| 12 | unique_urls | Distinct URLs |
| 13 | avg_url_length | Mean URL length |
| 14 | request_rate | Requests per second |
| 15 | ip_entropy | IP distribution entropy |
| 16 | url_entropy | URL distribution entropy |

## Model Architecture

```
Input (16)
    ↓
Dense(64) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(32) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(16) → BatchNorm → ReLU → Dropout(0.3)
    ↓
Dense(1) → Sigmoid
    ↓
Output (0=normal, 1=attack)
```

## Files

```
ddos/
├── config.py        # Configuration
├── features.py      # Feature extraction
├── model.py         # MLP model
├── train.py         # Training script
├── prepare_data.py  # Dataset preparation
├── requirements.txt # Dependencies
└── models/          # Saved models
    ├── apache_ddos_model.pt   # PyTorch
    ├── apache_ddos_model.pts  # TorchScript
    └── scaler.joblib          # Feature scaler
```

## API (Demo)

```bash
# Start services
cd demo/simple
docker-compose up -d

# Insert log
curl -X POST http://localhost:8000/log \
  -H "Content-Type: application/json" \
  -d '{"log": "192.168.1.1 - - [09/Dec/2025:10:00:00 +0000] \"GET /index.html HTTP/1.1\" 200 1234"}'

# Predict
curl -X POST http://localhost:8000/predict
```

## References

1. **Dataset**: [WebAttack-CVSSMetrics](https://huggingface.co/datasets/chYassine/WebAttack-CVSSMetrics)
2. **Paper**: Le, V.H., Zhang, H. (2022). Log-based Anomaly Detection with Deep Learning. ICSE'22.
