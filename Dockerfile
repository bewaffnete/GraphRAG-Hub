FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy all files needed for the build
# We need pyproject.toml, README.md, and the src directory
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install the project and all its dependencies
RUN pip install --no-cache-dir ".[all,mcp]"

# Install the project itself in editable mode or just re-run install to ensure scripts are linked
RUN pip install -e .

ENTRYPOINT ["graph-rag-mcp"]
