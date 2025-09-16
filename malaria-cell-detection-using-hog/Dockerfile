FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy only what is needed, explicitly
COPY requirements.txt ./
COPY app.py ./
COPY cell_images.zip ./
COPY extract.py ./
COPY start.sh ./
COPY malaria_svm_hog_model.joblib ./
COPY label_encoder.joblib ./
COPY examples/ ./examples/

# Set environment variables for Streamlit and Matplotlib config/cache
ENV STREAMLIT_HOME=/app/.streamlit
ENV MPLCONFIGDIR=/app/.config/matplotlib
ENV HOME=/app

RUN mkdir -p /app/.streamlit /app/.config/matplotlib
RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x start.sh

# Fix permissions for all files and folders
RUN chmod -R 777 /app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["./start.sh"]