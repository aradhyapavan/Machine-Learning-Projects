# Use a slim Python base image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install minimal runtime dependency for scikit-learn (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for better layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Create a non-root user for security
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Expose port (Hugging Face provides PORT env var at runtime)
EXPOSE 7860

# Streamlit config: ensure it binds to 0.0.0.0 and uses provided PORT
ENV PORT=7860

# Default command for Hugging Face Spaces (Docker): bind to $PORT
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0
