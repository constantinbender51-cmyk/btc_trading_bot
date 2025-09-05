FROM python:3.11-slim

WORKDIR /app

# Install system deps (pandas numpy need build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran liblapack-dev pkg-config && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default command = worker (can be overridden in Railway)
CMD ["python", "main.py"]
