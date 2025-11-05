FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Install system deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a venv and install requirements there (keeps things isolated)
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . /app

# Expose dev port
EXPOSE 8000

# Default command: run migrations then run server
CMD ["/opt/venv/bin/python", "app.py", "runserver", "0.0.0.0:8000"]
