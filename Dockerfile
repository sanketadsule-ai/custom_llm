# 1. Use Python 3.12+ for significantly improved async performance
FROM python:3.12-slim-bookworm

# 2. Set environment variables to optimize Python for containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONOPTIMIZE=1

WORKDIR /app

# 3. CRITICAL: Install build tools. 
# uvloop and orjson are C-extensions; they need these to compile correctly.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Use a cache-efficient way to install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of your app code
COPY . .

# 6. Optimized Production Command
# We use the 'exec' form [] and include performance flags
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--workers", "1", \
     "--no-access-log"]