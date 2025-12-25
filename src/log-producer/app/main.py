import os
import random
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from app.middleware import LoggingMiddleware
from app.kafka_producer import KafkaLogProducer

# Initialize Kafka producer
kafka_producer = KafkaLogProducer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    await kafka_producer.start()
    yield
    # Shutdown
    await kafka_producer.stop()


app = FastAPI(
    title="Log Producer API",
    description="A demo API that generates logs for the real-time log analysis system",
    version=os.getenv("APP_VERSION", "1.0.0"),
    lifespan=lifespan
)

# Add logging middleware
app.add_middleware(LoggingMiddleware, kafka_producer=kafka_producer)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "kafka_connected": kafka_producer.is_connected,
        "app_name": os.getenv("APP_NAME", "log-producer"),
        "version": os.getenv("APP_VERSION", "1.0.0")
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to Log Producer API", "docs": "/docs"}


@app.get("/api/users")
async def get_users(page: int = 1, limit: int = 10):
    """Get list of users (mock)"""
    users = [
        {"id": i, "name": f"User {i}", "email": f"user{i}@example.com"}
        for i in range((page - 1) * limit + 1, page * limit + 1)
    ]
    return {"users": users, "page": page, "limit": limit, "total": 100}


@app.get("/api/users/{user_id}")
async def get_user(user_id: int):
    """Get user by ID (mock)"""
    if user_id <= 0 or user_id > 100:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": user_id,
        "name": f"User {user_id}",
        "email": f"user{user_id}@example.com"
    }


@app.get("/api/products")
async def get_products(category: str = None, page: int = 1, limit: int = 10):
    """Get list of products (mock)"""
    products = [
        {
            "id": i,
            "name": f"Product {i}",
            "price": round(random.uniform(10, 1000), 2),
            "category": category or "general"
        }
        for i in range((page - 1) * limit + 1, page * limit + 1)
    ]
    return {"products": products, "page": page, "limit": limit}


@app.get("/api/orders")
async def get_orders(status: str = None):
    """Get list of orders (mock)"""
    statuses = ["pending", "processing", "shipped", "delivered"]
    orders = [
        {
            "id": i,
            "status": status or random.choice(statuses),
            "total": round(random.uniform(50, 500), 2)
        }
        for i in range(1, 11)
    ]
    return {"orders": orders}


@app.get("/api/slow")
async def slow_endpoint():
    """Slow endpoint for testing (simulates delay)"""
    import asyncio
    delay = random.uniform(0.5, 2.0)
    await asyncio.sleep(delay)
    return {"message": "Slow response", "delay_seconds": round(delay, 2)}


@app.get("/api/error")
async def error_endpoint():
    """Endpoint that returns server error"""
    raise HTTPException(status_code=500, detail="Internal server error (simulated)")


@app.get("/api/not-found")
async def not_found_endpoint():
    """Endpoint that returns 404"""
    raise HTTPException(status_code=404, detail="Resource not found (simulated)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
