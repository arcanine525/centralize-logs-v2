import asyncio
import json
import socket
from typing import Optional
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError
import os


class KafkaLogProducer:
    """Async Kafka producer for sending logs"""

    def __init__(
        self,
        bootstrap_servers: list = None,
        topic: str = None
    ):
        self.bootstrap_servers = bootstrap_servers or os.getenv(
            "KAFKA_BOOTSTRAP_SERVERS", "kafka:9092"
        ).split(",")
        self.topic = topic or os.getenv("KAFKA_TOPIC", "web-logs")
        self.producer: Optional[AIOKafkaProducer] = None
        self.hostname = socket.gethostname()
        self._is_connected = False

    async def start(self):
        """Initialize and start Kafka producer"""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                compression_type="snappy",
                max_batch_size=16384,
                linger_ms=10,
                acks=1,
            )
            await self.producer.start()
            self._is_connected = True
            print(f"Kafka producer connected to {self.bootstrap_servers}")
        except Exception as e:
            print(f"Failed to connect to Kafka: {e}")
            self._is_connected = False

    async def stop(self):
        """Stop Kafka producer"""
        if self.producer:
            await self.producer.stop()
            self._is_connected = False
            print("Kafka producer stopped")

    async def send_log(self, log_entry: dict) -> bool:
        """Send log entry to Kafka topic"""
        if not self._is_connected or not self.producer:
            return False

        try:
            log_entry['hostname'] = self.hostname
            await self.producer.send(
                topic=self.topic,
                value=log_entry,
                key=log_entry.get('client_ip', '').encode('utf-8')
            )
            return True
        except KafkaError as e:
            print(f"Failed to send log: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        return self._is_connected
