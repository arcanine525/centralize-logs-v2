"""
Kafka Consumer for DDoS Detection

Consumes logs from Kafka topic and performs real-time DDoS detection.
Runs as a background thread alongside the FastAPI server.
"""

import os
import json
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List
from collections import deque

# Kafka Consumer Config
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "web-logs")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "ddos-detector-group")
KAFKA_AUTO_OFFSET_RESET = os.getenv("KAFKA_AUTO_OFFSET_RESET", "latest")

# Processing Config
BATCH_SIZE = int(os.getenv("KAFKA_BATCH_SIZE", 100))
BATCH_TIMEOUT_MS = int(os.getenv("KAFKA_BATCH_TIMEOUT_MS", 5000))
WINDOW_SIZE = int(os.getenv("TIME_WINDOW", 60))

# Global state
consumer_thread: Optional[threading.Thread] = None
is_running = False
stats = {
    "consumed": 0,
    "processed": 0,
    "ddos_detected": 0,
    "normal": 0,
    "errors": 0,
    "last_message_time": None
}

# Sliding window buffer (keeps logs for feature extraction)
log_buffer: deque = deque(maxlen=10000)


def parse_kafka_message(msg_value: dict) -> Optional[Dict]:
    """Parse Kafka message to log format expected by feature extraction."""
    try:
        # JSON format from log-producer
        return {
            "ip": msg_value.get("client_ip", "0.0.0.0"),
            "timestamp": datetime.fromisoformat(msg_value.get("timestamp", datetime.now().isoformat())),
            "method": msg_value.get("method", "GET").upper(),
            "url": msg_value.get("path", "/"),
            "status": int(msg_value.get("status_code", 200)),
            "bytes": int(msg_value.get("response_size", 0))
        }
    except Exception as e:
        print(f"Error parsing Kafka message: {e}")
        return None


def get_window_logs(current_time: datetime, window_seconds: int = 60) -> List[Dict]:
    """Get logs within the time window for feature extraction."""
    cutoff_time = current_time.timestamp() - window_seconds
    return [
        log for log in log_buffer
        if log["timestamp"].timestamp() > cutoff_time
    ]


def process_batch(batch: List[dict], es_client, predict_fn, extract_features_fn, es_index: str):
    """Process a batch of Kafka messages."""
    global stats

    for msg in batch:
        try:
            parsed = parse_kafka_message(msg)
            if not parsed:
                stats["errors"] += 1
                continue

            # Add to sliding window buffer
            log_buffer.append(parsed)

            # Get window logs for feature extraction
            window_logs = get_window_logs(parsed["timestamp"], WINDOW_SIZE)

            # Extract features and predict
            features = extract_features_fn(window_logs)
            prob, status = predict_fn(features)

            # Store result in Elasticsearch
            doc = {
                "log": json.dumps(msg),
                "timestamp": parsed["timestamp"].isoformat(),
                "client_ip": parsed["ip"],
                "method": parsed["method"],
                "path": parsed["url"],
                "status_code": parsed["status"],
                "status": status,
                "probability": float(prob),
                "window_logs": len(window_logs)
            }
            es_client.index(index=es_index, document=doc)

            # Update stats
            stats["processed"] += 1
            if status == "DDOS":
                stats["ddos_detected"] += 1
            else:
                stats["normal"] += 1

        except Exception as e:
            print(f"Error processing message: {e}")
            stats["errors"] += 1


def kafka_consumer_loop(es_client, predict_fn, extract_features_fn, es_index: str):
    """Main Kafka consumer loop."""
    global is_running, stats

    try:
        from kafka import KafkaConsumer
    except ImportError:
        print("kafka-python not installed. Run: pip install kafka-python")
        return

    print(f"Starting Kafka consumer...")
    print(f"  Bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"  Topic: {KAFKA_TOPIC}")
    print(f"  Group ID: {KAFKA_GROUP_ID}")
    print(f"  Batch size: {BATCH_SIZE}")

    consumer = None
    retry_count = 0
    max_retries = 10

    while is_running and retry_count < max_retries:
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS.split(","),
                group_id=KAFKA_GROUP_ID,
                auto_offset_reset=KAFKA_AUTO_OFFSET_RESET,
                enable_auto_commit=True,
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                max_poll_interval_ms=300000,
                session_timeout_ms=30000,
            )
            print(f"Connected to Kafka successfully!")
            retry_count = 0  # Reset on success

            batch = []
            last_batch_time = time.time()

            while is_running:
                # Poll for messages
                messages = consumer.poll(timeout_ms=1000, max_records=BATCH_SIZE)

                for topic_partition, msgs in messages.items():
                    for msg in msgs:
                        stats["consumed"] += 1
                        stats["last_message_time"] = datetime.now().isoformat()
                        batch.append(msg.value)

                # Process batch if full or timeout
                current_time = time.time()
                if len(batch) >= BATCH_SIZE or (batch and current_time - last_batch_time > BATCH_TIMEOUT_MS / 1000):
                    process_batch(batch, es_client, predict_fn, extract_features_fn, es_index)
                    batch = []
                    last_batch_time = current_time

        except Exception as e:
            retry_count += 1
            print(f"Kafka consumer error (retry {retry_count}/{max_retries}): {e}")
            time.sleep(5)

    if consumer:
        consumer.close()
    print("Kafka consumer stopped")


def start_consumer(es_client, predict_fn, extract_features_fn, es_index: str):
    """Start the Kafka consumer in a background thread."""
    global consumer_thread, is_running

    if consumer_thread and consumer_thread.is_alive():
        print("Kafka consumer already running")
        return False

    is_running = True
    consumer_thread = threading.Thread(
        target=kafka_consumer_loop,
        args=(es_client, predict_fn, extract_features_fn, es_index),
        daemon=True
    )
    consumer_thread.start()
    print("Kafka consumer thread started")
    return True


def stop_consumer():
    """Stop the Kafka consumer."""
    global is_running
    is_running = False
    print("Stopping Kafka consumer...")


def get_stats() -> dict:
    """Get consumer statistics."""
    return {
        **stats,
        "is_running": is_running,
        "buffer_size": len(log_buffer),
        "kafka_topic": KAFKA_TOPIC,
        "kafka_servers": KAFKA_BOOTSTRAP_SERVERS
    }
