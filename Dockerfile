FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch
RUN pip install --no-cache-dir \
    torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu

# onnxruntime for Linux — installed separately from requirements.txt
# because the Windows version (1.16.3) doesn't exist on Linux
# 1.18.1 is the latest stable Linux-compatible version
RUN pip install --no-cache-dir onnxruntime==1.18.1

# All other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source code
COPY api/          ./api/
COPY embeddings/   ./embeddings/
COPY models/       ./models/
COPY cache/        ./cache/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]