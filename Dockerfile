# Dockerfile for Predictive Occupancy Modeling API
#
# This image installs the project as a Python package along with its
# dependencies and exposes a FastAPI server for training and
# querying occupancy models.  Jupyter and plotting utilities have
# been removed to keep the image lightweight.  The container
# automatically runs the API under uvicorn on port 8000.

FROM python:3.12-slim

# Install system packages required for building some Python wheels
RUN apt-get update -qq && \
    apt-get install --no-install-recommends -y \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the entire repository into the container
COPY . /app

# Upgrade pip and install the project in editable mode.  FastAPI and
# uvicorn are installed via the project's dependencies declared in
# pyproject.toml.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Expose the API port
EXPOSE 8000

# Default command: launch the FastAPI server via uvicorn.  Uvicorn
# will read the app object from src.api and listen on all interfaces.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]