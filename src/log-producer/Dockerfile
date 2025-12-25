FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for python-snappy
RUN apt-get update && apt-get install -y \
    build-essential \
    libsnappy-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
