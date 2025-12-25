import time
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timezone


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests to Kafka"""

    def __init__(self, app, kafka_producer):
        super().__init__(app)
        self.kafka_producer = kafka_producer

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())
        start_time = time.time()

        # Extract request details
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "")
        referer = request.headers.get("referer", "")

        # Process request
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise
        finally:
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000

            # Create log entry
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client_ip": client_ip,
                "user_agent": user_agent,
                "referer": referer,
                "status_code": status_code,
                "response_time_ms": round(response_time_ms, 2),
                "request_id": request_id,
            }

            # Send log to Kafka asynchronously
            await self.kafka_producer.send_log(log_entry)

        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP (handle proxies)"""
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        if request.client:
            return request.client.host
        return "unknown"
