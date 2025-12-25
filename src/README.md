# Centralize Logs - Real-time Log Analysis System

Hệ thống phân tích log máy chủ theo thời gian thực với khả năng phát hiện tấn công DDoS bằng Machine Learning.

## 🏗️ Kiến Trúc

```
Log Producer → Kafka → Logstash → Elasticsearch → Kibana
     :8000       :9092     :9600       :9200        :5601
                   ↓
              DDoS API (Kafka Consumer)
                :28000
```

**DDoS Detection**: Sử dụng MLP model (TorchScript) với 16 features từ Apache logs.

## 📁 Cấu Trúc Project

```
src/
├── docker-compose.yml          # Orchestration (7 services)
├── .env                        # Environment variables
├── configs/
│   ├── logstash/
│   │   ├── logstash.conf      # Pipeline configuration
│   │   └── logstash.yml       # Logstash settings
│   └── elasticsearch/
│       └── web-logs-template.json
├── log-producer/              # Python web server
│   ├── app/
│   │   ├── main.py
│   │   ├── middleware.py
│   │   └── kafka_producer.py
│   ├── requirements.txt
│   └── Dockerfile
├── scripts/
│   ├── attack_simulation.py       # DoS simulation
│   └── import_kibana_dashboards.sh # Dashboard import
└── data/                      # Persistent volumes

BKMS_BigData2025/demo/simple/  # DDoS Detection API
├── app.py                     # FastAPI + Kafka consumer
├── kafka_consumer.py          # Background Kafka consumer
├── Dockerfile
└── models/                    # ML models
    ├── apache_ddos_model.pts  # TorchScript model
    └── scaler.joblib          # Feature scaler
```

## 🚀 Quick Start

### 1. Khởi động hệ thống

```bash
cd src

# Start all services (including DDoS API)
docker compose up -d --build

# Xem logs
docker compose logs -f
```

### 2. Kiểm tra services

| Service | URL | Mục đích |
|---------|-----|----------|
| **Kibana** | http://localhost:5601 | Dashboard & Visualization |
| **Elasticsearch** | http://localhost:9200 | Search API |
| **Log Producer** | http://localhost:8000 | Demo API |
| **DDoS API** | http://localhost:28000 | ML Detection |
| **Logstash** | http://localhost:9600 | Pipeline Monitoring |

### 3. Import Kibana Dashboards

```bash
./scripts/import_kibana_dashboards.sh
```

### 4. Tạo test traffic

```bash
# Normal traffic
curl http://localhost:8000/api/users
curl http://localhost:8000/api/products
curl http://localhost:8000/health

# Simulate DoS attack
python scripts/attack_simulation.py --mode dos --duration 60 --rate 50
```

### 5. Kiểm tra DDoS Detection

```bash
# Health check
curl http://localhost:28000/health

# Kafka consumer stats
curl http://localhost:28000/kafka/stats

# View detections
curl "http://localhost:9200/ddos-logs/_search?q=status:DDOS&pretty"
```

## 🤖 DDoS Detection API

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health + Kafka stats |
| `/kafka/start` | POST | Start Kafka consumer |
| `/kafka/stop` | POST | Stop Kafka consumer |
| `/kafka/stats` | GET | Consumer statistics |
| `/predict` | POST | Batch predict |

DDoS API consume trực tiếp từ Kafka topic `web-logs`. Auto-start khi `KAFKA_ENABLED=true`.

## 🛠️ Development

### Rebuild services

```bash
docker compose build ddos-api
docker compose up -d ddos-api
```

### View Kafka topics

```bash
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092
```

### Check Elasticsearch

```bash
curl http://localhost:9200/_cat/indices?v
```

## 📊 Resource Requirements

**Development (MacBook 16GB):**
- Docker Desktop: 8GB RAM
- Total services: ~5GB RAM

**Production (GCP):**

| Tier | Machine Type | vCPU | RAM | Use Case |
|------|--------------|------|-----|----------|
| Demo | `e2-standard-4` | 4 | 16 GB | Testing |
| Prod | `e2-standard-8` | 8 | 32 GB | Production |

## ☁️ GCP Deployment

```bash
# Create VM
gcloud compute instances create log-analysis-demo \
  --machine-type=e2-standard-4 \
  --zone=asia-southeast1-b \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd

# Open firewall
gcloud compute firewall-rules create allow-log-demo \
  --allow=tcp:5601,tcp:8000,tcp:9200,tcp:28000 \
  --target-tags=http-server
```

## 🛑 Stop Services

```bash
docker compose down

# Remove volumes (clean data)
docker compose down -v
```
