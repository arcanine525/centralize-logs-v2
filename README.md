# Centralize Logs

Real-time log aggregation and analysis platform with ML-based DDoS detection.

## Features

- **Real-time Log Collection**: Capture HTTP logs from web servers via Kafka
- **Data Processing Pipeline**: Enrich logs with GeoIP, user agent parsing, and response categorization
- **Full-text Search**: Store and query logs in Elasticsearch
- **Visualization**: Kibana dashboards for log analysis and monitoring
- **DDoS Detection**: MLP neural network model detecting attacks from traffic patterns

## Architecture

```
Log Producer (FastAPI) → Kafka → Logstash → Elasticsearch → Kibana
      :8000              :9092     :9600        :9200         :5601
                           ↓
                    DDoS Detection API
                         :28000
```

| Service | Port | Description |
|---------|------|-------------|
| Log Producer | 8000 | Demo web server generating HTTP logs |
| Kafka | 9092 | Message broker for log ingestion |
| Logstash | 9600 | Log processing and enrichment |
| Elasticsearch | 9200 | Search and analytics engine |
| Kibana | 5601 | Visualization dashboards |
| DDoS API | 28000 | ML-based attack detection |

## Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for scripts)
- 8GB+ RAM recommended

## Quick Start

```bash
# Clone and start services
cd src
docker compose up -d --build

# Verify services are running
docker compose ps

# View logs
docker compose logs -f
```

### Health Checks

```bash
curl http://localhost:8000/health     # Log Producer
curl http://localhost:28000/health    # DDoS API
curl http://localhost:9200/_cluster/health  # Elasticsearch
```

### Generate Test Traffic

```bash
# Normal requests
curl http://localhost:8000/api/users
curl http://localhost:8000/api/products

# Simulate DDoS attack
python src/scripts/attack_simulation.py --mode dos --duration 60 --rate 50

# Normal traffic simulation
python src/scripts/attack_simulation.py --mode normal --duration 60 --rate 10
```

### View Results

- **Kibana**: http://localhost:5601
- **DDoS detections**: `curl "http://localhost:9200/ddos-logs/_search?q=status:DDOS&pretty"`

## Project Structure

```
├── src/                          # Main production system
│   ├── docker-compose.yml        # 7-service orchestration
│   ├── .env                      # Environment configuration
│   ├── configs/
│   │   ├── logstash/             # Pipeline configuration
│   │   └── elasticsearch/        # Index templates
│   ├── log-producer/             # FastAPI log generator
│   └── scripts/                  # Utility scripts
│
├── BKMS_BigData2025/
│   ├── demo/simple/              # DDoS Detection API
│   │   ├── app.py                # FastAPI + Kafka consumer
│   │   └── models/               # Trained ML models
│   └── ddos/                     # Model training code
│
└── technical-docs/               # Documentation (Vietnamese)
```

## DDoS Detection Model

**Architecture**: MLP (16 → 64 → 32 → 16 → 1) with BatchNorm, ReLU, and Dropout

**Input Features** (extracted from 60-second windows):

| Feature | Description |
|---------|-------------|
| request_count | Total requests in window |
| unique_ips | Distinct IP addresses |
| requests_per_ip | Average requests per IP |
| get_ratio / post_ratio | HTTP method distribution |
| status_2xx/4xx/5xx_ratio | Response status distribution |
| ip_entropy | IP address distribution entropy |
| url_entropy | URL path distribution entropy |
| request_rate | Requests per second |

**Training Data**: [WebAttack-CVSSMetrics](https://huggingface.co/datasets/chYassine/WebAttack-CVSSMetrics) (18,842 labeled logs)

**Performance**: 99.75% accuracy, 99.50% recall

## API Reference

### Log Producer (port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with Kafka status |
| `/api/users` | GET | List users (mock) |
| `/api/products` | GET | List products (mock) |
| `/api/orders` | GET | List orders (mock) |
| `/api/slow` | GET | Slow endpoint (0.5-2s) |
| `/api/error` | GET | Error endpoint (500) |

### DDoS API (port 28000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health + Kafka consumer stats |
| `/kafka/start` | POST | Start Kafka consumer |
| `/kafka/stop` | POST | Stop Kafka consumer |
| `/kafka/stats` | GET | Consumer statistics |
| `/predict` | POST | Batch prediction |

## Configuration

Key environment variables in `src/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BOOTSTRAP_SERVERS` | kafka:9092 | Kafka connection |
| `KAFKA_TOPIC` | web-logs | Topic for log messages |
| `ELASTICSEARCH_HOSTS` | http://elasticsearch:9200 | ES connection |
| `THRESHOLD` | 0.5 | DDoS detection threshold |
| `TIME_WINDOW` | 60 | Feature extraction window (seconds) |

## Development

### Rebuild a service

```bash
docker compose build ddos-api
docker compose up -d ddos-api
```

### View Kafka messages

```bash
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic web-logs --from-beginning
```

### Query Elasticsearch

```bash
# List indices
curl http://localhost:9200/_cat/indices?v

# Search logs
curl 'http://localhost:9200/web-logs-*/_search?pretty&size=10'
```

### Train the model

```bash
cd BKMS_BigData2025
pip install -r ddos/requirements.txt
python -m ddos.prepare_data --output dataset/train.csv
python -m ddos.train --data dataset/train.csv --epochs 100
```

## Deployment

### Resource Requirements

| Environment | CPU | RAM | Disk |
|-------------|-----|-----|------|
| Development | 4 cores | 8 GB | 20 GB |
| Production | 8 cores | 32 GB | 100 GB |

### GCP Deployment

```bash
gcloud compute instances create log-analysis \
  --machine-type=e2-standard-4 \
  --zone=asia-southeast1-b \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

gcloud compute firewall-rules create allow-log-ports \
  --allow=tcp:5601,tcp:8000,tcp:9200,tcp:28000
```

## Stop Services

```bash
docker compose down        # Stop containers
docker compose down -v     # Stop and remove volumes
```

## References

- [WebAttack-CVSSMetrics Dataset](https://huggingface.co/datasets/chYassine/WebAttack-CVSSMetrics)
- Le, V.H., Zhang, H. (2022). Log-based Anomaly Detection with Deep Learning. ICSE'22.

## License

MIT
