# Start with official Echidna image (includes Echidna + Slither + solc)
FROM ghcr.io/crytic/echidna/echidna:latest

# Install Python 3.12 (available in Ubuntu 24.04) and system dependencies
# Including JDK 21 for Certora Prover
RUN apt-get update && \
    apt-get install -y \
        python3.12 \
        python3.12-venv \
        python3-pip \
        libssl-dev \
        libffi-dev \
        build-essential \
        git \
        openjdk-21-jdk && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create symlink so 'python3' points to python3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --break-system-packages --no-cache-dir -r requirements.txt

# Install Mythril
RUN pip3 install --break-system-packages mythril

# Install Certora CLI for formal verification
RUN pip3 install --break-system-packages certora-cli

# Copy application code
COPY . .

# Ensure solc is configured (already in Echidna image, but set our version)
RUN python3 -m solc_select install 0.8.24 && \
    python3 -m solc_select use 0.8.24

# Create necessary directories for logs and temp files
RUN mkdir -p /opt/render/project/data /tmp/users.db

# Verify all tools are available
RUN echo "========== Tool Verification ==========" && \
    echidna --version && \
    slither --version && \
    myth version && \
    python3 --version && \
    solc --version && \
    certoraRun --version && \
    echo "========== All Tools Ready ==========="

# Expose port for FastAPI
EXPOSE 10000

# Start command (FastAPI with gunicorn)
CMD python3 -m gunicorn main:app --workers ${WEB_CONCURRENCY:-2} --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:10000 --timeout 300 --graceful-timeout 120 --log-level info --preload