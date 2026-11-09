# Use lightweight official Python runtime
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=8501

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install CPU-optimized PyTorch and TorchVision (significantly reduces image size)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-cache Hugging Face Vision Transformer weights during build so container starts instantly offline
RUN python -c "from transformers import ViTForImageClassification; ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=2, ignore_mismatched_sizes=True)"

# Copy application code and model artifacts
COPY .streamlit/ .streamlit/
COPY main.py model_utils.py fetch_satellite.py metrics.txt ./
COPY best_model.pth ./

# Expose Streamlit default port
EXPOSE 8501

# Add container health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8501}/_stcore/health || exit 1

# Start Streamlit application, dynamically binding to $PORT (compatible with Render, Railway, Cloud Run, AWS)
CMD ["sh", "-c", "streamlit run main.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"]

