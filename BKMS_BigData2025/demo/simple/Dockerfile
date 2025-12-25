FROM python:3.11-slim

WORKDIR /app

# Install other packages from PyPI (elasticsearch<9 for ES 8.x compatibility)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    "elasticsearch>=8,<9" \
    numpy \
    joblib \
    scikit-learn \
    kafka-python

# Install PyTorch CPU-only from PyTorch index
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY app.py .
COPY kafka_consumer.py .

EXPOSE 8000

CMD ["python", "app.py"]

