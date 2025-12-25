# DDoS Detection Demo

Minimal production demo using ONNX Runtime (~50MB vs PyTorch ~2GB).

## Files

```
simple/
├── app.py              # FastAPI + ONNX Runtime
├── docker-compose.yml
├── Dockerfile
├── simulate.py         # Import logs & trigger
└── README.md
```

## Quick Start

```bash
# 1. Export model to ONNX (from project root)
cd ../..
python -m ddos.export

# 2. Start services
cd demo/simple
docker-compose up -d

# 3. Wait for ES (~30s)
curl http://localhost:9200

# 4. Import your logs and predict
python simulate.py --file ../../dataset/access.log
```

## Production Size

| Package | Size |
|---------|------|
| torch | ~2GB |
| onnxruntime | ~50MB |

## Usage

```bash
# Import from access.log file
python simulate.py --file ../../dataset/access.log

# Import with limit
python simulate.py --file ../../dataset/access.log --limit 1000

# Sample data (for testing)
python simulate.py              # normal traffic
python simulate.py --attack     # DDoS simulation

# Just trigger prediction (logs already in ES)
python simulate.py --trigger

# View stats
python simulate.py --stats

# Clear and reimport
python simulate.py --clear --file ../../dataset/access.log
```

## How It Works

```
1. Logs stored in ES (from file import, Filebeat, etc.):
   {"log": "192.168.1.1 - - [15/Jan/2024:10:00:00] GET /...", "timestamp": "..."}

2. Trigger /predict (manually, cron, Kafka, etc.):
   - Queries logs without status
   - Groups by 60s time window
   - Extracts features
   - Model predicts

3. ES document updated:
   {"log": "...", "timestamp": "...", "status": "DDOS", "probability": 0.85}
```

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /log` | Insert log + predict window (streaming) |
| `POST /predict` | Predict all unpredicted logs (batch/trigger) |
| `GET /health` | Health check |

## Trigger Scenarios

### 1. Manual trigger
```bash
curl -X POST http://localhost:8000/predict
```

### 2. Cron job (every minute)
```bash
* * * * * curl -X POST http://localhost:8000/predict
```

### 3. After log import (in your pipeline)
```python
# After Filebeat/Logstash imports logs to ES
requests.post("http://localhost:8000/predict")
```

### 4. Streaming (real-time)
```python
for line in tail_log_file():
    requests.post("http://localhost:8000/log", json={"log": line})
```

## View Results

```bash
# All DDOS detections
curl 'http://localhost:9200/logs/_search?q=status:DDOS&pretty'

# Count by status
curl 'http://localhost:9200/logs/_search?pretty' -H 'Content-Type: application/json' -d '{
  "size": 0,
  "aggs": {"status": {"terms": {"field": "status"}}}
}'
```
