# Elasticsearch Query Commands

## 🧪 Test Full Pipeline (Log Producer → Kafka → Logstash → ES)

### Step 1: Kiểm tra services đang chạy

```bash
docker compose ps
```

### Step 2: Gửi request tới Log Producer để tạo logs

```bash
# Gửi 1 request
curl http://localhost:8000/health

# Gửi nhiều requests
for i in {1..10}; do curl -s http://localhost:8000/api/users > /dev/null; done
echo "Sent 10 requests"

# Gửi request 404
curl http://localhost:8000/api/not-found
```

### Step 3: Kiểm tra log trong Kafka topic

```bash
# List topics
docker exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Đọc messages từ topic (xem 5 messages mới nhất)
docker exec kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic web-logs \
  --from-beginning \
  --max-messages 5
```

### Step 4: Kiểm tra Logstash đang xử lý

```bash
# Xem logs của Logstash
docker logs logstash --tail 20

# Kiểm tra pipeline stats
curl -s "http://localhost:9600/_node/stats/pipelines?pretty" | grep -A5 '"events"'
```

### Step 5: Kiểm tra logs đã vào Elasticsearch

```bash
# Đợi vài giây rồi query
sleep 3

# Đếm số logs
curl -s "http://localhost:9200/web-logs-*/_count?pretty"

# Xem 5 logs mới nhất
curl -s -X GET "http://localhost:9200/web-logs-*/_search?pretty&size=5" \
  -H 'Content-Type: application/json' \
  -d '{"query": {"match_all": {}}, "sort": [{"@timestamp": "desc"}]}'
```

### 🔄 One-liner Test (tất cả trong 1 command)

```bash
echo "=== Sending requests ===" && \
for i in {1..5}; do curl -s http://localhost:8000/api/users > /dev/null; done && \
echo "Sent 5 requests, waiting 5s..." && sleep 5 && \
echo "=== Checking ES ===" && \
curl -s "http://localhost:9200/web-logs-*/_count?pretty"
```

---

## Kiểm tra Indices

```bash
curl -s "http://localhost:9200/_cat/indices?v"
```

## Query tất cả logs (5 logs mới nhất)

```bash
curl -s -X GET "http://localhost:9200/web-logs-*/_search?pretty&size=5" \
  -H 'Content-Type: application/json' \
  -d '{"query": {"match_all": {}}, "sort": [{"@timestamp": "desc"}]}'
```

## Query logs theo status code

```bash
# Logs có status 404
curl -s -X GET "http://localhost:9200/web-logs-*/_search?pretty&size=10" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {"term": {"status_code": 404}},
    "sort": [{"@timestamp": "desc"}]
  }'

# Logs có status 5xx (server errors)
curl -s -X GET "http://localhost:9200/web-logs-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {"range": {"status_code": {"gte": 500, "lt": 600}}},
    "sort": [{"@timestamp": "desc"}]
  }'
```

## Query logs theo IP

```bash
curl -s -X GET "http://localhost:9200/web-logs-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {"term": {"client_ip": "192.168.1.100"}},
    "sort": [{"@timestamp": "desc"}]
  }'
```

## Query logs theo path

```bash
curl -s -X GET "http://localhost:9200/web-logs-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {"wildcard": {"path": "/api/*"}},
    "sort": [{"@timestamp": "desc"}]
  }'
```

## Aggregation: Top 10 IPs

```bash
curl -s -X GET "http://localhost:9200/web-logs-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "size": 0,
    "aggs": {
      "top_ips": {
        "terms": {"field": "client_ip", "size": 10}
      }
    }
  }'
```

## Aggregation: Status code distribution

```bash
curl -s -X GET "http://localhost:9200/web-logs-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "size": 0,
    "aggs": {
      "status_codes": {
        "terms": {"field": "status_code"}
      }
    }
  }'
```

## Query logs trong 15 phút gần nhất

```bash
curl -s -X GET "http://localhost:9200/web-logs-*/_search?pretty" \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {
      "range": {
        "@timestamp": {"gte": "now-15m", "lte": "now"}
      }
    },
    "sort": [{"@timestamp": "desc"}]
  }'
```

## Đếm tổng số logs

```bash
curl -s "http://localhost:9200/web-logs-*/_count?pretty"
```

## Xóa index (cẩn thận!)

```bash
curl -X DELETE "http://localhost:9200/web-logs-2024.12.10"
```
