FROM python:3.13-slim

WORKDIR /app

# System deps (important for scientific / ML libs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy dependency files first (for caching)
COPY pyproject.toml uv.lock* ./

# Install dependencies with uv
RUN uv pip install --system .

# Copy full project
COPY . .

# Streamlit port
EXPOSE 8501

# Streamlit config (important for Docker)
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

# Run app
CMD ["streamlit", "run", "app.py"]