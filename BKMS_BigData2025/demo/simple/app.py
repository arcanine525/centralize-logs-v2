"""
DDoS Detection API with Kafka Consumer Support

Features:
- TorchScript model for inference
- Option 1: HTTP endpoint for Logstash integration
- Option 2: Kafka consumer for high-performance streaming
"""

import re
import os
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from elasticsearch import Elasticsearch

# ============================================================================
# CONFIG
# ============================================================================

ES_HOST = os.getenv("ES_HOST", "localhost")
ES_PORT = int(os.getenv("ES_PORT", 9200))
ES_INDEX = os.getenv("ES_INDEX", "logs")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/model.pts")
TIME_WINDOW = int(os.getenv("TIME_WINDOW", 60))
THRESHOLD = float(os.getenv("THRESHOLD", 0.5))
INPUT_DIM = int(os.getenv("INPUT_DIM", 16))  # 16 Apache log features

# Kafka Consumer Config
KAFKA_ENABLED = os.getenv("KAFKA_ENABLED", "false").lower() == "true"
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "web-logs")

LOG_PATTERN = re.compile(
    r'(\d+\.\d+\.\d+\.\d+)\s+-\s+-\s+\[([^\]]+)\]\s+"(\w+)\s+([^"]+)[^"]*"\s+(\d+)\s+(\d+|-)'
)

print(f"ES_HOST: {ES_HOST}")
print(f"ES_PORT: {ES_PORT}")
print(f"ES_INDEX: {ES_INDEX}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"TIME_WINDOW: {TIME_WINDOW}")
print(f"THRESHOLD: {THRESHOLD}")
print(f"INPUT_DIM: {INPUT_DIM}")
print(f"KAFKA_ENABLED: {KAFKA_ENABLED}")
print(f"KAFKA_BOOTSTRAP_SERVERS: {KAFKA_BOOTSTRAP_SERVERS}")
print(f"KAFKA_TOPIC: {KAFKA_TOPIC}")

# ============================================================================
# MODEL (TorchScript + Scaler)
# ============================================================================

model = None
scaler = None
SCALER_PATH = os.getenv("SCALER_PATH", "/app/models/scaler.joblib")

def load_model():
    global model, scaler
    import torch
    import joblib

    if os.path.exists(MODEL_PATH):
        model = torch.jit.load(MODEL_PATH, map_location='cpu')
        model.eval()
        print(f"TorchScript model loaded: {MODEL_PATH}")
    else:
        print(f"Model not found: {MODEL_PATH}")

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded: {SCALER_PATH}")
    else:
        print(f"Scaler not found: {SCALER_PATH} (using unscaled features)")

def predict(features: np.ndarray) -> tuple:
    """Run model inference on extracted features."""
    if model is None:
        return 0.0, "UNKNOWN"

    import torch

    # Pad or truncate to match model input dimension
    if len(features) < INPUT_DIM:
        features = np.pad(features, (0, INPUT_DIM - len(features)))
    elif len(features) > INPUT_DIM:
        features = features[:INPUT_DIM]

    # Scale features (important - model trained on scaled data)
    if scaler is not None:
        features = scaler.transform(features.reshape(1, -1)).flatten()

    with torch.no_grad():
        x = torch.FloatTensor(features.astype(np.float32)).unsqueeze(0)
        prob = model(x).item()

    return prob, "DDOS" if prob >= THRESHOLD else "NORMAL"

# ============================================================================
# ELASTICSEARCH
# ============================================================================

es: Optional[Elasticsearch] = None

def get_es():
    global es
    if es is None:
        es = Elasticsearch([f"http://{ES_HOST}:{ES_PORT}"])
    return es

def ensure_index():
    e = get_es()
    try:
        if not e.indices.exists(index=ES_INDEX):
            e.indices.create(index=ES_INDEX, mappings={
                "properties": {
                    "log": {"type": "text"},
                    "timestamp": {"type": "date"},
                    "status": {"type": "keyword"},
                    "probability": {"type": "float"}
                }
            })
    except Exception as ex:
        print(f"Index check/create: {ex}")

# ============================================================================
# HELPERS
# ============================================================================

def parse_log(raw: str) -> Optional[Dict]:
    m = LOG_PATTERN.match(raw.strip())
    if not m:
        return None
    ip, ts, method, url, status, bytes_ = m.groups()
    try:
        ts = datetime.strptime(ts.split()[0], '%d/%b/%Y:%H:%M:%S')
    except:
        ts = datetime.now()
    return {"ip": ip, "timestamp": ts, "method": method.upper(),
            "url": url, "status": int(status), "bytes": int(bytes_) if bytes_ != '-' else 0}

def get_window_range(ts: datetime) -> tuple:
    return ts - timedelta(seconds=TIME_WINDOW), ts

def entropy(items: List) -> float:
    """Calculate Shannon entropy of a list."""
    if not items:
        return 0.0
    from collections import Counter
    counts = Counter(items)
    total = len(items)
    return -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)

def extract_features(logs: List[Dict]) -> np.ndarray:
    """
    Extract features from logs based on ICSE'22 paper methodology.
    Features designed for HTTP anomaly detection.
    """
    if not logs:
        return np.zeros(16, dtype=np.float32)

    n = len(logs)
    ips = [l['ip'] for l in logs]
    methods = [l['method'] for l in logs]
    statuses = [l['status'] for l in logs]
    urls = [l['url'] for l in logs]
    bytes_list = [l['bytes'] for l in logs]

    unique_ips = len(set(ips))
    unique_urls = len(set(urls))

    return np.array([
        # Volume features
        n,                                      # 1. request_count
        unique_ips,                             # 2. unique_ips
        n / max(unique_ips, 1),                 # 3. requests_per_ip (high = suspicious)

        # Method features
        len(set(methods)),                      # 4. unique_methods
        methods.count('GET') / n,               # 5. get_ratio
        methods.count('POST') / n,              # 6. post_ratio

        # Response features
        np.mean(bytes_list) if bytes_list else 0,  # 7. avg_bytes
        sum(bytes_list),                        # 8. total_bytes
        sum(200 <= s < 300 for s in statuses) / n,  # 9. status_2xx_ratio
        sum(400 <= s < 500 for s in statuses) / n,  # 10. status_4xx_ratio
        sum(s >= 500 for s in statuses) / n,   # 11. status_5xx_ratio

        # URL features
        unique_urls,                            # 12. unique_urls
        np.mean([len(u) for u in urls]),        # 13. avg_url_length

        # Rate features
        n / TIME_WINDOW,                        # 14. request_rate (req/sec)

        # Entropy features (distribution uniformity)
        entropy(ips),                           # 15. ip_entropy (low = single source attack)
        entropy(urls),                          # 16. url_entropy (low = repetitive attack)
    ], dtype=np.float32)

# ============================================================================
# API
# ============================================================================

app = FastAPI(title="DDoS Detection", version="2.0", description="HTTP + Kafka Consumer modes")

class LogReq(BaseModel):
    log: str

@app.on_event("startup")
async def startup():
    load_model()
    ensure_index()

    # Start Kafka consumer if enabled
    if KAFKA_ENABLED:
        try:
            from kafka_consumer import start_consumer
            start_consumer(get_es(), predict, extract_features, ES_INDEX)
            print("Kafka consumer started automatically")
        except Exception as e:
            print(f"Failed to start Kafka consumer: {e}")

@app.on_event("shutdown")
async def shutdown():
    if KAFKA_ENABLED:
        try:
            from kafka_consumer import stop_consumer
            stop_consumer()
        except:
            pass

@app.get("/health")
async def health():
    kafka_status = None
    if KAFKA_ENABLED:
        try:
            from kafka_consumer import get_stats
            kafka_status = get_stats()
        except:
            kafka_status = {"error": "kafka_consumer not available"}

    return {
        "ok": True,
        "model": model is not None,
        "threshold": THRESHOLD,
        "kafka_enabled": KAFKA_ENABLED,
        "kafka": kafka_status
    }

@app.post("/log")
async def insert_log(req: LogReq):
    """Insert log + predict using sliding window."""
    parsed = parse_log(req.log)
    if not parsed:
        raise HTTPException(400, "Invalid log")

    e = get_es()
    doc_id = e.index(index=ES_INDEX, document={"log": req.log, "timestamp": parsed['timestamp'].isoformat()})['_id']

    ws, we = get_window_range(parsed['timestamp'])
    hits = e.search(index=ES_INDEX, query={
        "range": {"timestamp": {"gte": ws.isoformat(), "lte": we.isoformat()}}
    }, size=10000)['hits']['hits']

    logs = [p for h in hits if (p := parse_log(h['_source']['log']))]
    prob, status = predict(extract_features(logs))

    e.update(index=ES_INDEX, id=doc_id, doc={"status": status, "probability": prob})
    return {"id": doc_id, "status": status, "probability": round(prob, 4), "window_logs": len(logs)}

MAX_ES_SIZE = 10000  # Elasticsearch default max

@app.get("/logs/unpredicted")
async def get_unpredicted(limit: int = 1000):
    """Get all logs without prediction status."""
    e = get_es()
    size = min(limit, MAX_ES_SIZE)
    hits = e.search(index=ES_INDEX, query={
        "bool": {"must_not": {"exists": {"field": "status"}}}
    }, size=size, sort=[{"timestamp": "asc"}])['hits']['hits']

    return {
        "count": len(hits),
        "logs": [{"id": h['_id'], **h['_source']} for h in hits]
    }

@app.get("/logs/all")
async def get_all_logs(limit: int = 1000):
    """Get all logs with their status."""
    e = get_es()
    size = min(limit, MAX_ES_SIZE)
    hits = e.search(index=ES_INDEX, size=size, sort=[{"timestamp": "desc"}])['hits']['hits']

    return {
        "count": len(hits),
        "logs": [{"id": h['_id'], **h['_source']} for h in hits]
    }

@app.post("/predict")
async def predict_all(batch_size: int = 500):
    """Predict ALL unpredicted logs in batches and write back to ES."""
    e = get_es()
    batch_size = min(batch_size, MAX_ES_SIZE)

    total_processed = 0
    total_ddos = 0
    total_normal = 0

    # Loop until no more unpredicted logs
    while True:
        # Get batch of unpredicted logs
        hits = e.search(index=ES_INDEX, query={
            "bool": {"must_not": {"exists": {"field": "status"}}}
        }, size=batch_size, sort=[{"timestamp": "asc"}])['hits']['hits']

        if not hits:
            break  # No more unpredicted logs

        print(f"Processing batch of {len(hits)} logs...", flush=True)

        # Process each log in batch
        for h in hits:
            parsed = parse_log(h['_source']['log'])
            if not parsed:
                continue

            # Get window logs for feature extraction
            ws, we = get_window_range(parsed['timestamp'])
            wh = e.search(index=ES_INDEX, query={
                "range": {"timestamp": {"gte": ws.isoformat(), "lte": we.isoformat()}}
            }, size=1000)['hits']['hits']

            logs = [p for x in wh if (p := parse_log(x['_source']['log']))]
            prob, status = predict(extract_features(logs))

            # Write back to ES
            e.update(index=ES_INDEX, id=h['_id'], doc={"status": status, "probability": prob})

            if status == "DDOS":
                total_ddos += 1
            else:
                total_normal += 1
            total_processed += 1

        print(f"  Processed: {total_processed}, DDOS: {total_ddos}, Normal: {total_normal}", flush=True)

    return {
        "processed": total_processed,
        "ddos": total_ddos,
        "normal": total_normal
    }

# ============================================================================
# KAFKA CONSUMER CONTROL ENDPOINTS
# ============================================================================

@app.post("/kafka/start")
async def start_kafka():
    """Manually start Kafka consumer."""
    try:
        from kafka_consumer import start_consumer, get_stats
        success = start_consumer(get_es(), predict, extract_features, ES_INDEX)
        return {"started": success, "stats": get_stats()}
    except ImportError:
        raise HTTPException(500, "kafka-python not installed")
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/kafka/stop")
async def stop_kafka():
    """Stop Kafka consumer."""
    try:
        from kafka_consumer import stop_consumer, get_stats
        stop_consumer()
        return {"stopped": True, "stats": get_stats()}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/kafka/stats")
async def kafka_stats():
    """Get Kafka consumer statistics."""
    try:
        from kafka_consumer import get_stats
        return get_stats()
    except ImportError:
        return {"error": "kafka_consumer not available", "kafka_enabled": KAFKA_ENABLED}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
